import 'dart:convert';

import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import '../models/road_data.dart';

class NotificationService {
  NotificationService._internal();
  static final NotificationService _instance = NotificationService._internal();

  factory NotificationService() => _instance;

  final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();

  static const String channelId = 'road_alerts_channel';
  static const String channelName = 'Road Alerts';
  static const String channelDescription = 'Real-time road safety notifications';
  
  static const String accidentChannelId = 'accident_alerts_channel';
  static const String accidentChannelName = 'Accident Alerts';
  static const String accidentChannelDescription = 'Accident detection notifications';
  
  static const String emergencyChannelId = 'emergency_alerts_channel';
  static const String emergencyChannelName = 'Emergency Vehicle Alerts';
  static const String emergencyChannelDescription = 'Emergency vehicle detection notifications';

  bool _initialized = false;
  Function(String?)? _onNotificationTap;

  Future<void> initialize({Function(String?)? onNotificationTap}) async {
    if (_initialized) {
      _onNotificationTap = onNotificationTap;
      return;
    }

    _onNotificationTap = onNotificationTap;

    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const DarwinInitializationSettings iosSettings =
        DarwinInitializationSettings(
      requestSoundPermission: true,
      requestBadgePermission: true,
      requestAlertPermission: true,
    );

    final InitializationSettings settings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _notifications.initialize(
      settings,
      onDidReceiveNotificationResponse: (NotificationResponse response) {
        _onNotificationTap?.call(response.payload);
      },
      onDidReceiveBackgroundNotificationResponse: notificationTapBackground,
    );

    final androidPlugin = _notifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>();

    if (androidPlugin != null) {
      // Create default channel
      const AndroidNotificationChannel defaultChannel = AndroidNotificationChannel(
        channelId,
        channelName,
        description: channelDescription,
        importance: Importance.high,
        playSound: true,
        enableVibration: true,
      );
      await androidPlugin.createNotificationChannel(defaultChannel);

      // Create accident channel with custom sound support
      // To add custom sound: Place accident.mp3 in android/app/src/main/res/raw/
      // Then uncomment the sound line below
      final AndroidNotificationChannel accidentChannel = AndroidNotificationChannel(
        accidentChannelId,
        accidentChannelName,
        description: accidentChannelDescription,
        importance: Importance.max,
        playSound: true,
        enableVibration: true,
        enableLights: true,
        // Uncomment below to use custom sound file:
        // sound: const RawResourceAndroidNotificationSound('accident'),
      );
      await androidPlugin.createNotificationChannel(accidentChannel);

      // Create emergency channel with custom sound support
      // To add custom sound: Place emergency.mp3 in android/app/src/main/res/raw/
      // Then uncomment the sound line below
      final AndroidNotificationChannel emergencyChannel = AndroidNotificationChannel(
        emergencyChannelId,
        emergencyChannelName,
        description: emergencyChannelDescription,
        importance: Importance.max,
        playSound: true,
        enableVibration: true,
        enableLights: true,
        // Uncomment below to use custom sound file:
        // sound: const RawResourceAndroidNotificationSound('emergency'),
      );
      await androidPlugin.createNotificationChannel(emergencyChannel);
    }

    _initialized = true;
  }

  Future<void> requestPermissions() async {
    // Request Firebase Messaging permissions if available
    try {
      await FirebaseMessaging.instance.requestPermission(
        alert: true,
        badge: true,
        sound: true,
        announcement: false,
        criticalAlert: true,
        provisional: false,
      );
    } catch (e) {
      // Firebase not initialized - local notifications will still work
      debugPrint('Firebase Messaging permissions skipped: $e');
    }

    // Android notification permissions are handled automatically by the system
    // when the app first shows a notification (Android 13+)
  }

  Future<void> showLocalNotification({
    required String title,
    required String body,
    String? payload,
    String? customChannelId,
    String? customChannelName,
  }) async {
    final channelIdToUse = customChannelId ?? channelId;
    final channelNameToUse = customChannelName ?? channelName;
    
    final AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      channelIdToUse,
      channelNameToUse,
      channelDescription: customChannelId == accidentChannelId
          ? accidentChannelDescription
          : customChannelId == emergencyChannelId
              ? emergencyChannelDescription
              : channelDescription,
      importance: customChannelId != null ? Importance.max : Importance.high,
      priority: customChannelId != null ? Priority.max : Priority.high,
      playSound: true,
      enableVibration: true,
      vibrationPattern: customChannelId != null
          ? Int64List.fromList([0, 500, 200, 500, 200, 500])
          : null,
      styleInformation: const BigTextStyleInformation(''),
    );

    const DarwinNotificationDetails iosDetails =
        DarwinNotificationDetails(presentAlert: true, presentSound: true);

    final notificationDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      notificationDetails,
      payload: payload,
    );
  }

  Future<void> showRoadAlert({
    required RoadData road,
    required String type,
  }) async {
    final isAccident = type == 'accident';
    final title = isAccident
        ? '🚨 ACCIDENT on ${road.roadName}'
        : '🚨 EMERGENCY VEHICLE near ${road.roadName}';
    final body = isAccident
        ? 'Avoid this route immediately and follow alternate paths.'
        : 'Please make way for the emergency vehicle immediately.';

    await showLocalNotification(
      title: title,
      body: body,
      payload: jsonEncode({'roadId': road.roadId}),
      customChannelId: isAccident ? accidentChannelId : emergencyChannelId,
      customChannelName: isAccident ? accidentChannelName : emergencyChannelName,
    );
  }

  Future<void> showRemoteNotification(RemoteMessage message) async {
    final notification = message.notification;
    final title = notification?.title ?? 'Road Eye Alert';
    final body = notification?.body ??
        (message.data['body'] ?? 'Check road updates in the app.');
    final payload = message.data.isNotEmpty ? jsonEncode(message.data) : null;
    await showLocalNotification(title: title, body: body, payload: payload);
  }

  void handleNotificationTap(String? payload) {
    _onNotificationTap?.call(payload);
  }
}

@pragma('vm:entry-point')
void notificationTapBackground(NotificationResponse response) {
  NotificationService().handleNotificationTap(response.payload);
}


