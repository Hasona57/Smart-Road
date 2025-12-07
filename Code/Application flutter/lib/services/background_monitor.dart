import 'dart:async';
import 'dart:convert';
import 'package:workmanager/workmanager.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';
import '../models/road_data.dart';
import 'notification_service.dart';

@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((task, inputData) async {
    try {
      // Try to initialize Firebase (optional)
      try {
        await Firebase.initializeApp();
      } catch (e) {
        debugPrint('Firebase initialization skipped in background: $e');
      }
      
      await NotificationService().initialize();
      
      final monitor = BackgroundMonitor();
      await monitor.checkForAlerts();
      return true;
    } catch (e) {
      debugPrint('Background task error: $e');
      return false;
    }
  });
}

class BackgroundMonitor {
  static const String baseUrl =
      'https://smart-traffic-system-4ac4b-default-rtdb.firebaseio.com';
  
  static const String _lastAccidentStatesKey = 'last_accident_states';
  static const String _lastEmergencyStatesKey = 'last_emergency_states';

  Future<void> checkForAlerts() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final url = Uri.parse('$baseUrl/roads.json');
      
      final response = await http
          .get(url, headers: {'Content-Type': 'application/json'})
          .timeout(const Duration(seconds: 10));

      if (response.statusCode != 200) {
        return;
      }

      final dynamic data = json.decode(response.body);
      if (data == null || data is! Map) {
        return;
      }

      // Load previous states
      final lastAccidentStatesJson = prefs.getString(_lastAccidentStatesKey);
      final lastEmergencyStatesJson = prefs.getString(_lastEmergencyStatesKey);
      
      final Map<String, bool> lastAccidentStates = lastAccidentStatesJson != null
          ? Map<String, bool>.from(json.decode(lastAccidentStatesJson))
          : {};
      final Map<String, bool> lastEmergencyStates = lastEmergencyStatesJson != null
          ? Map<String, bool>.from(json.decode(lastEmergencyStatesJson))
          : {};

      final Map<String, bool> currentAccidentStates = {};
      final Map<String, bool> currentEmergencyStates = {};

      // Process each road
      for (final entry in data.entries) {
        final value = entry.value;
        if (value is Map) {
          final road = RoadData.fromRealtimeJson(
            Map<String, dynamic>.from(value),
          );
          
          final roadId = road.roadId;
          final prevAccident = lastAccidentStates[roadId] ?? false;
          final prevEmergency = lastEmergencyStates[roadId] ?? false;

          // Check for new accidents
          if (road.hasAccident && !prevAccident) {
            await NotificationService().showRoadAlert(
              road: road,
              type: 'accident',
            );
          }

          // Check for new emergency vehicles
          if (road.hasEmergencyVehicle && !prevEmergency) {
            await NotificationService().showRoadAlert(
              road: road,
              type: 'emergency',
            );
          }

          currentAccidentStates[roadId] = road.hasAccident;
          currentEmergencyStates[roadId] = road.hasEmergencyVehicle;
        }
      }

      // Save current states
      await prefs.setString(
        _lastAccidentStatesKey,
        json.encode(currentAccidentStates),
      );
      await prefs.setString(
        _lastEmergencyStatesKey,
        json.encode(currentEmergencyStates),
      );
    } catch (e) {
      debugPrint('Background monitor error: $e');
    }
  }
}

