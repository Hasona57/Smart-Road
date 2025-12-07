import 'dart:async';
import 'dart:convert';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:share_plus/share_plus.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:http/http.dart' as http;
import 'package:workmanager/workmanager.dart';

import 'models/road_data.dart';
import 'services/notification_service.dart';
import 'services/road_cache_manager.dart';
import 'services/background_monitor.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();
final NotificationService notificationService = NotificationService();

@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  try {
    await Firebase.initializeApp();
    await notificationService.initialize();
    await notificationService.showRemoteNotification(message);
  } catch (e) {
    // Firebase not configured - skip background notification
    debugPrint('Background message handler error: $e');
  }
}

class FirebaseRealtimeService {
  static const String baseUrl =
      'https://smart-traffic-system-4ac4b-default-rtdb.firebaseio.com';

  final RoadCacheManager _cacheManager = RoadCacheManager();
  final http.Client _client = http.Client();

  Future<Map<String, RoadData>> fetchAllRoads() async {
    final url = Uri.parse('$baseUrl/roads.json');
    try {
      final response = await _client
          .get(url, headers: {'Content-Type': 'application/json'})
          .timeout(const Duration(seconds: 5));

      if (response.statusCode != 200) {
        throw Exception('Failed to load roads (${response.statusCode})');
      }

      final dynamic data = json.decode(response.body);
      if (data == null || data is! Map) {
        throw Exception('Invalid data format');
      }

      final Map<String, RoadData> roads = {};
      data.forEach((key, value) {
        if (value is Map) {
          roads[key.toString()] =
              RoadData.fromRealtimeJson(Map<String, dynamic>.from(value));
        }
      });

      if (roads.isNotEmpty) {
        await _cacheManager.saveRoads(roads);
      }

      return roads;
          } catch (e) {
      debugPrint('Firebase fetch error: $e');
      rethrow;
    }
  }

  Future<RoadData?> fetchRoad(String roadId) async {
    final url = Uri.parse('$baseUrl/roads/$roadId.json');
    try {
      final response = await _client
          .get(url, headers: {'Content-Type': 'application/json'})
          .timeout(const Duration(seconds: 5));

      if (response.statusCode != 200) {
        throw Exception('Failed to load road');
      }

      final dynamic data = json.decode(response.body);
      if (data == null || data is! Map) return null;

      final road =
          RoadData.fromRealtimeJson(Map<String, dynamic>.from(data));
      await _cacheManager.saveRoad(road);
      return road;
    } catch (e) {
      debugPrint('Firebase fetch error: $e');
      return null;
    }
  }

  Future<CachedRoadBundle?> loadCachedRoads() => _cacheManager.loadRoads();

  Future<RoadData?> loadCachedRoad(String roadId) =>
      _cacheManager.loadRoad(roadId);

  Stream<RoadData> getRoadStream(String roadId) {
    final controller = StreamController<RoadData>();
    Timer? timer;

    Future<void> fetchData() async {
      final road = await fetchRoad(roadId);
      if (road != null && !controller.isClosed) {
        controller.add(road);
      } else {
        final cached = await _cacheManager.loadRoad(roadId);
        if (cached != null && !controller.isClosed) {
          controller.add(cached);
        }
      }
    }

    timer = Timer.periodic(const Duration(seconds: 3), (_) => fetchData());
    fetchData();

    controller.onCancel = () {
      timer?.cancel();
    };

    return controller.stream;
  }
}

// AI Analysis

class RoadAnalysis {
  static String getSuggestion(RoadData data) {
    if (data.hasAccident) {
      return 'EMERGENCY: Accident Detected! Avoid this route immediately. ${data.accidentDetails ?? "Emergency services are on the way."}';
    }
    
    if (data.hasEmergencyVehicle) {
      return 'EMERGENCY: Emergency Vehicle Approaching! Clear the way and expect delays.';
    }

    bool isTrafficBad = data.trafficState == 'Heavy Congestion' ||
        (data.trafficLightStatus == 'Red' && data.travelTimeMinutes > 15);
    bool isPollutionHigh = data.pollutionPpm > 15.0;
    bool isTravelTimeExcessive = data.travelTimeMinutes > 25;

    if (isTrafficBad && isPollutionHigh && isTravelTimeExcessive) {
      return 'CRITICAL: Seek Alternative Route Immediately. High congestion, severe pollution, and excessive travel time detected.';
    } else if (isTrafficBad && isPollutionHigh) {
      return 'CRITICAL: Seek Alternative Route. High congestion and severe pollution detected.';
    } else if (isTrafficBad && isTravelTimeExcessive) {
      return 'WARNING: High Traffic & Long Delays. Consider alternative routes to save time.';
    } else if (isPollutionHigh && isTravelTimeExcessive) {
      return 'CAUTION: High Pollution & Long Travel Time. Consider a greener, faster alternative.';
    } else if (isTrafficBad) {
      return 'WARNING: High Traffic Detected. Expect significant delays. Consider alternatives.';
    } else if (isPollutionHigh) {
      return 'CAUTION: High Pollution Levels. Proceed with care or choose a greener route.';
    } else if (isTravelTimeExcessive) {
      return 'NOTICE: Longer Travel Time Expected. You may want to consider faster routes.';
    } else if (data.trafficState == 'Clear' && 
               data.travelTimeMinutes < 10 && 
               data.pollutionPpm < 10) {
      return 'OPTIMAL: This is the Best Route Now! Clear traffic, low travel time, and clean air.';
    } else if (data.trafficState == 'Clear' && data.travelTimeMinutes < 10) {
      return 'GOOD: Clear traffic and low travel time. Route is recommended.';
    } else {
      return 'NORMAL: The road state is currently acceptable. Safe to proceed.';
    }
  }

  static int getSafetyScore(RoadData data) {
    int score = 100;
    if (data.hasAccident) score -= 50;
    if (data.hasEmergencyVehicle) score -= 20;
    if (data.trafficState == 'Heavy Congestion') score -= 15;
    if (data.pollutionPpm > 15.0) score -= 10;
    if (data.travelTimeMinutes > 25) score -= 10;
    if (data.trafficLightStatus == 'Red') score -= 5;
    return score.clamp(0, 100);
  }
}

// Maps Service

class MapsService {
  static Future<void> openDirections(String destination, {double? lat, double? lon}) async {
    String url;
    
    if (lat != null && lon != null) {
      url = 'https://www.google.com/maps/dir/?api=1&destination=$lat,$lon';
    } else {
      url = 'https://www.google.com/maps/dir/?api=1&destination=${Uri.encodeComponent(destination)}';
    }
    
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } else {
      throw 'Could not launch Google Maps';
    }
  }
}


Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize notifications (works without Firebase)
  await notificationService.initialize(onNotificationTap: _handleNotificationTap);
  await notificationService.requestPermissions();

  // Try to initialize Firebase (optional - app works with REST API without it)
  try {
    await Firebase.initializeApp();
    FirebaseMessaging.onBackgroundMessage(firebaseMessagingBackgroundHandler);
    await FirebaseMessaging.instance.setAutoInitEnabled(true);
    await FirebaseMessaging.instance.setForegroundNotificationPresentationOptions(
      alert: true,
      badge: true,
      sound: true,
    );
    await FirebaseMessaging.instance.subscribeToTopic('road_alerts');

    FirebaseMessaging.onMessage.listen(
      (message) => notificationService.showRemoteNotification(message),
    );

    FirebaseMessaging.onMessageOpenedApp.listen(
      (message) => _handleNotificationNavigation(message.data['roadId']),
    );

    final initialMessage = await FirebaseMessaging.instance.getInitialMessage();
    if (initialMessage != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _handleNotificationNavigation(initialMessage.data['roadId']);
      });
    }
  } catch (e) {
    // Firebase not configured - app will work with REST API only
    // Local notifications will still work
    debugPrint('Firebase initialization skipped: $e');
  }
  
  // Initialize background monitoring
  try {
    await Workmanager().initialize(
      callbackDispatcher,
      isInDebugMode: false,
    );
    // Register periodic task to check for alerts every 15 minutes
    // Note: Minimum frequency is 15 minutes on Android
    await Workmanager().registerPeriodicTask(
      'road-alerts-check',
      'roadAlertsCheck',
      frequency: const Duration(minutes: 15),
    );
  } catch (e) {
    debugPrint('Workmanager initialization skipped: $e');
  }
  
  runApp(const RoadEyeApp());
}

class RoadEyeApp extends StatelessWidget {
  const RoadEyeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Road Eye',
      debugShowCheckedModeBanner: false,
      navigatorKey: navigatorKey,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F7FA),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF007BFF),
          brightness: Brightness.light,
        ),
        cardTheme: CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const RoadSelectionPage(),
    );
  }
}

// Road Selection Page

class RoadSelectionPage extends StatefulWidget {
  const RoadSelectionPage({super.key});

  @override
  State<RoadSelectionPage> createState() => _RoadSelectionPageState();
}

class _RoadSelectionPageState extends State<RoadSelectionPage> {
  final FirebaseRealtimeService _service = FirebaseRealtimeService();
  Map<String, RoadData> _roads = {};
  bool _isLoading = true;
  String _searchQuery = '';
  bool _isConnected = false;
  String? _errorMessage;
  String _sortBy = 'name'; // name, safety, time, pollution
  DateTime? _lastUpdated;
  bool _isStaleData = false;
  final Map<String, bool> _lastAccidentState = {};
  final Map<String, bool> _lastEmergencyState = {};
  Timer? _autoRefreshTimer;
  final bool _autoRefreshEnabled = true;
  final int _refreshIntervalSeconds = 3;

  @override
  void initState() {
    super.initState();
    _loadRoads();
    _startAutoRefresh();
  }

  void _startAutoRefresh() {
    _autoRefreshTimer?.cancel();
    if (_autoRefreshEnabled) {
      _autoRefreshTimer = Timer.periodic(
        Duration(seconds: _refreshIntervalSeconds),
        (_) => _loadRoads(),
      );
    }
  }

  @override
  void dispose() {
    _autoRefreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadRoads() async {
      setState(() {
      _isLoading = true;
      _errorMessage = null;
    });
    try {
      final roads = await _service.fetchAllRoads();
      setState(() {
        _roads = roads;
        _isLoading = false;
        _isConnected = true;
        _isStaleData = false;
        _lastUpdated = DateTime.now();
        _errorMessage = null;
      });
      _handleAlertNotifications(roads, fromCache: false);
    } catch (e) {
      final cached = await _service.loadCachedRoads();
      if (cached != null) {
        setState(() {
          _roads = cached.roads;
          _isLoading = false;
          _isConnected = false;
          _isStaleData = true;
          _lastUpdated = cached.timestamp;
          _errorMessage =
              'Offline mode. Showing saved data from ${_formatTimestamp(cached.timestamp)}';
        });
      } else {
        setState(() {
          _isLoading = false;
          _isConnected = false;
          _errorMessage =
              'Unable to load road data. Please check your connection.';
          _roads = {};
        });
      }
    }
  }

  List<MapEntry<String, RoadData>> get _filteredRoads {
    List<MapEntry<String, RoadData>> roads = _roads.entries.toList();
    
    if (_searchQuery.isNotEmpty) {
      roads = roads.where((entry) => entry.value.roadName
          .toLowerCase()
          .contains(_searchQuery.toLowerCase())).toList();
    }
    
    roads.sort((a, b) {
      switch (_sortBy) {
        case 'safety':
          return RoadAnalysis.getSafetyScore(b.value)
              .compareTo(RoadAnalysis.getSafetyScore(a.value));
        case 'time':
          return a.value.travelTimeMinutes.compareTo(b.value.travelTimeMinutes);
        case 'pollution':
          return a.value.pollutionPpm.compareTo(b.value.pollutionPpm);
        default:
          return a.value.roadName.compareTo(b.value.roadName);
      }
    });
    
    return roads;
  }

  void _handleAlertNotifications(Map<String, RoadData> roads,
      {required bool fromCache}) {
    if (fromCache) return;
    roads.forEach((key, road) {
      final prevAccident = _lastAccidentState[key] ?? false;
      final prevEmergency = _lastEmergencyState[key] ?? false;

      if (road.hasAccident && !prevAccident) {
        notificationService.showRoadAlert(road: road, type: 'accident');
      }
      if (road.hasEmergencyVehicle && !prevEmergency) {
        notificationService.showRoadAlert(road: road, type: 'emergency');
      }

      _lastAccidentState[key] = road.hasAccident;
      _lastEmergencyState[key] = road.hasEmergencyVehicle;
    });
  }

  String _formatTimestamp(DateTime timestamp) =>
      DateFormat('MMM d, h:mm a').format(timestamp);

  Widget _buildOfflineBanner() {
    if (!_isStaleData || _lastUpdated == null) {
      return const SizedBox.shrink();
    }
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
      decoration: BoxDecoration(
        color: Colors.orange.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.orangeAccent),
      ),
      child: Row(
        children: [
          const Icon(Icons.wifi_off, color: Colors.orange),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Offline mode – showing saved data from ${_formatTimestamp(_lastUpdated!)}',
              style: const TextStyle(
                color: Colors.orange,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF007BFF),
              Color(0xFF0056D6),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
        children: [
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(
                            Icons.route,
                            color: Colors.white,
                            size: 28,
                          ),
                        ),
                        const SizedBox(width: 16),
                        const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Road Eye',
                  style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 28,
                    fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                'Select a road to view details',
                                style: TextStyle(
                                  color: Colors.white70,
                                  fontSize: 14,
                                ),
                              ),
                            ],
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: _isConnected 
                                ? Colors.green.withValues(alpha: 0.3)
                                : Colors.red.withValues(alpha: 0.3),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                _isConnected ? Icons.cloud_done : Icons.cloud_off,
                                color: Colors.white,
                                size: 16,
                              ),
                              const SizedBox(width: 4),
                              Text(
                                _isConnected ? 'Online' : 'Offline',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
                          onPressed: _loadRoads,
                          tooltip: 'Refresh',
                        ),
                        IconButton(
                          icon: const Icon(Icons.settings, color: Colors.white),
            onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => const SettingsPage(),
                              ),
                            );
                          },
                          tooltip: 'Settings',
          ),
        ],
      ),
                    const SizedBox(height: 12),
                    _buildOfflineBanner(),
                    const SizedBox(height: 12),
                    Row(
        children: [
                        Expanded(
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(12),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withValues(alpha: 0.05),
                                  blurRadius: 10,
                                  offset: const Offset(0, 4),
                                ),
                              ],
                            ),
                            child: TextField(
                              onChanged: (value) => setState(() => _searchQuery = value),
                              decoration: InputDecoration(
                                hintText: 'Search roads...',
                                prefixIcon: const Icon(Icons.search, color: Color(0xFF007BFF)),
                                border: InputBorder.none,
                                contentPadding: const EdgeInsets.all(16),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        PopupMenuButton<String>(
                          icon: Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(12),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withValues(alpha: 0.05),
                                  blurRadius: 10,
                                  offset: const Offset(0, 4),
                                ),
                              ],
                            ),
                            child: const Icon(Icons.sort, color: Color(0xFF007BFF)),
                          ),
                          onSelected: (value) => setState(() => _sortBy = value),
                          itemBuilder: (context) => [
                            const PopupMenuItem(
                              value: 'name',
                              child: Row(
                                children: [
                                  Icon(Icons.sort_by_alpha, size: 20),
                                  SizedBox(width: 8),
                                  Text('Sort by Name'),
                                ],
                              ),
                            ),
                            const PopupMenuItem(
                              value: 'safety',
                              child: Row(
                                children: [
                                  Icon(Icons.shield, size: 20),
                                  SizedBox(width: 8),
                                  Text('Sort by Safety'),
                                ],
                              ),
                            ),
                            const PopupMenuItem(
                              value: 'time',
                              child: Row(
                                children: [
                                  Icon(Icons.access_time, size: 20),
                                  SizedBox(width: 8),
                                  Text('Sort by Time'),
                                ],
                              ),
                            ),
                            const PopupMenuItem(
                              value: 'pollution',
                              child: Row(
                                children: [
                                  Icon(Icons.air, size: 20),
                                  SizedBox(width: 8),
                                  Text('Sort by Pollution'),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              Expanded(
                child: Container(
                  decoration: const BoxDecoration(
                    color: Color(0xFFF5F7FA),
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(30),
                      topRight: Radius.circular(30),
                    ),
                  ),
                  child: _isLoading
                      ? Center(
            child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const CircularProgressIndicator(),
                const SizedBox(height: 20),
                              Text(
                                'Loading roads...',
                  style: TextStyle(
                                  color: Colors.grey.shade600,
                                  fontSize: 16,
                                ),
                              ),
                            ],
                          ),
                        )
                      : _filteredRoads.isEmpty
                          ? Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    _errorMessage != null 
                                        ? Icons.error_outline 
                                        : Icons.route_outlined,
                                    size: 64,
                                    color: Colors.grey.shade400,
                                  ),
                                  const SizedBox(height: 16),
                                  Text(
                                    _errorMessage ?? 
                                    (_searchQuery.isEmpty
                                        ? 'No roads available'
                                        : 'No roads found'),
                                    style: TextStyle(
                                      color: Colors.grey.shade600,
                                      fontSize: 18,
                                    ),
                                    textAlign: TextAlign.center,
                                  ),
                                  if (_errorMessage != null) ...[
                                    const SizedBox(height: 12),
                                    ElevatedButton.icon(
                                      onPressed: _loadRoads,
                                      icon: const Icon(Icons.refresh),
                                      label: const Text('Retry'),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: const Color(0xFF007BFF),
                                        foregroundColor: Colors.white,
                                      ),
                                    ),
                                  ],
                                ],
                              ),
                            )
                          : Column(
                              children: [
                                if (_filteredRoads.isNotEmpty)
                                  Container(
                                    width: double.infinity,
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 16,
                                      vertical: 12,
                                    ),
                                    color: Colors.white,
                                    child: Row(
                                      children: [
                                        Icon(
                                          Icons.info_outline,
                                          size: 18,
                                          color: Colors.grey.shade600,
                                        ),
                                        const SizedBox(width: 8),
                                        Text(
                                          '${_filteredRoads.length} road${_filteredRoads.length != 1 ? 's' : ''} found',
                                          style: TextStyle(
                                            color: Colors.grey.shade600,
                                            fontSize: 14,
                                            fontWeight: FontWeight.w500,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                Expanded(
                                  child: RefreshIndicator(
                                    onRefresh: _loadRoads,
                                    child: ListView.builder(
                                      padding: const EdgeInsets.all(16),
                                      itemCount: _filteredRoads.length,
                                      itemBuilder: (context, index) {
                                        final entry = _filteredRoads[index];
                                        final road = entry.value;
                                        return _buildRoadCard(road);
                                      },
                                    ),
                                  ),
                                ),
                              ],
                            ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRoadCard(RoadData road) {
    final safetyScore = RoadAnalysis.getSafetyScore(road);
    Color statusColor = _getStatusColor(road);
    IconData statusIcon = _getStatusIcon(road);

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(milliseconds: 300),
      builder: (context, value, child) {
        return Transform.scale(
          scale: value,
          child: Opacity(
            opacity: value,
            child: child,
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => RoadDetailsPage(roadId: road.roadId),
                ),
              ).then((_) => _loadRoads());
            },
          borderRadius: BorderRadius.circular(16),
          child: Container(
      padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
              borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 10,
                  offset: const Offset(0, 4),
              ),
            ],
          ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
        children: [
                Row(
            children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: statusColor.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Icon(statusIcon, color: statusColor, size: 24),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
              Text(
                            road.roadName,
                style: const TextStyle(
                              fontSize: 18,
                  fontWeight: FontWeight.bold,
                              color: Color(0xFF1E293B),
                ),
              ),
                          const SizedBox(height: 4),
              Text(
                            road.trafficState,
                style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey.shade600,
                ),
              ),
            ],
                      ),
          ),
          Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: _getScoreColor(safetyScore).withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        '$safetyScore',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: _getScoreColor(safetyScore),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
            children: [
                    _buildInfoChip(
                      Icons.traffic,
                      road.trafficLightStatus,
                      _getLightColor(road.trafficLightStatus),
                    ),
                    const SizedBox(width: 8),
                    _buildInfoChip(
                      Icons.access_time,
                      '${road.travelTimeMinutes.toStringAsFixed(1)} min',
                      Colors.blue,
                    ),
                    const SizedBox(width: 8),
                    _buildInfoChip(
                      Icons.air,
                      '${road.pollutionPpm.toStringAsFixed(1)} ppm',
                      _getPollutionColor(road.pollutionPpm),
                    ),
                  ],
                ),
                if (road.hasEmergencyVehicle || road.hasAccident) ...[
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      if (road.hasEmergencyVehicle)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [Colors.orange.shade600, Colors.orange.shade800],
                            ),
                            borderRadius: BorderRadius.circular(10),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.orange.withValues(alpha: 0.6),
                                blurRadius: 4,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.local_police, color: Colors.white, size: 18),
                              const SizedBox(width: 6),
                              const Text(
                                'Emergency Vehicle',
                                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      if (road.hasAccident)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [Colors.red.shade600, Colors.red.shade800],
                            ),
                            borderRadius: BorderRadius.circular(10),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.red.withValues(alpha: 0.5),
                                blurRadius: 4,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.car_crash, color: Colors.white, size: 18),
                              const SizedBox(width: 6),
                              const Text(
                                'Accident',
                                style: TextStyle(
                color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
      ),
    );
  }

  Widget _buildInfoChip(IconData icon, String label, Color color) {
        return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: color,
            ),
              ),
            ],
          ),
    );
  }

  Color _getStatusColor(RoadData road) {
    if (road.hasAccident) return Colors.red;
    if (road.hasEmergencyVehicle) return Colors.orange;
    if (road.trafficState == 'Heavy Congestion') return Colors.red.shade700;
    if (road.trafficState == 'Moderate') return Colors.amber;
    return Colors.green;
  }

  IconData _getStatusIcon(RoadData road) {
    if (road.hasAccident) return Icons.warning;
    if (road.hasEmergencyVehicle) return Icons.local_police;
    if (road.trafficState == 'Heavy Congestion') return Icons.traffic;
    if (road.trafficState == 'Moderate') return Icons.slow_motion_video;
    return Icons.check_circle;
  }

  Color _getScoreColor(int score) {
    if (score >= 80) return Colors.green;
    if (score >= 60) return Colors.orange;
    return Colors.red;
  }

  Color _getLightColor(String status) {
    switch (status) {
      case 'Green': return Colors.green;
      case 'Yellow': return Colors.amber;
      case 'Red': return Colors.red;
      default: return Colors.grey;
    }
  }

  Color _getPollutionColor(double ppm) {
    if (ppm > 15.0) return Colors.purple.shade700;
    if (ppm > 10.0) return Colors.orange.shade700;
    return Colors.lightBlue.shade700;
  }
}

// Road Details Page

class RoadDetailsPage extends StatefulWidget {
  final String roadId;

  const RoadDetailsPage({super.key, required this.roadId});

  @override
  State<RoadDetailsPage> createState() => _RoadDetailsPageState();
}

class _RoadDetailsPageState extends State<RoadDetailsPage> {
  final FirebaseRealtimeService _service = FirebaseRealtimeService();
  RoadData? _roadData;
  String _suggestion = 'Loading...';
  StreamSubscription<RoadData>? _subscription;
  bool _isLoading = true;
  bool _isCachedData = false;
  DateTime? _cacheTimestamp;
  bool _previousAccident = false;
  bool _previousEmergency = false;

  @override
  void initState() {
    super.initState();
    _loadRoadData();
  }

  void _loadRoadData() {
    setState(() => _isLoading = true);
    _subscription?.cancel();
    _subscription = _service.getRoadStream(widget.roadId).listen(
      (data) {
        if (!mounted) return;
        final wasAccident = _previousAccident;
        final wasEmergency = _previousEmergency;
        _previousAccident = data.hasAccident;
        _previousEmergency = data.hasEmergencyVehicle;
              setState(() {
          _roadData = data;
          _suggestion = RoadAnalysis.getSuggestion(data);
          _isLoading = false;
          _isCachedData = data.isFromCache;
          _cacheTimestamp = data.cachedAt ?? data.lastUpdated;
        });
        if (!data.isFromCache) {
          if (data.hasAccident && !wasAccident) {
            notificationService.showRoadAlert(road: data, type: 'accident');
          } else if (data.hasEmergencyVehicle && !wasEmergency) {
            notificationService.showRoadAlert(road: data, type: 'emergency');
          }
        }
      },
      onError: (error) {
        () async {
          final cached = await _service.loadCachedRoad(widget.roadId);
          if (!mounted) return;
          if (cached != null) {
            setState(() {
              _roadData = cached;
              _suggestion = RoadAnalysis.getSuggestion(cached);
              _isLoading = false;
              _isCachedData = true;
              _cacheTimestamp = cached.cachedAt ?? cached.lastUpdated;
            });
    } else {
            setState(() {
              _isLoading = false;
              _suggestion = 'Error loading data';
            });
          }
        }();
      },
    );
  }

  String _formatTimestamp(DateTime timestamp) =>
      DateFormat('MMM d, h:mm a').format(timestamp);

  @override
  void dispose() {
    _subscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _isLoading || _roadData == null
          ? Container(
              decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
                  colors: [Color(0xFF007BFF), Color(0xFF0056D6)],
                ),
              ),
              child: SafeArea(
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const CircularProgressIndicator(color: Colors.white),
                      const SizedBox(height: 20),
                      const Text(
                        'Loading road data...',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            )
          : RefreshIndicator(
              onRefresh: () async {
                _loadRoadData();
              },
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                child: Column(
            children: [
                    _buildHeader(),
                    const SizedBox(height: 20),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: Column(
                        children: [
                          _buildQuickStats(),
          const SizedBox(height: 12),
          _buildCacheBanner(),
                          const SizedBox(height: 20),
                          _buildEmergencyBanner(),
                          const SizedBox(height: 20),
                          _buildRecommendationCard(),
                          const SizedBox(height: 20),
                          _buildSafetyScoreCard(),
                          const SizedBox(height: 20),
                          _buildMetricsGrid(),
                          const SizedBox(height: 20),
                          _buildDirectionsButton(),
                          const SizedBox(height: 20),
                          _buildLastUpdatedCard(),
                          const SizedBox(height: 20),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildQuickStats() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildStatItem(
            Icons.access_time,
            _roadData!.travelTimeMinutes.toStringAsFixed(1),
            'min',
            Colors.blue,
          ),
          Container(
            width: 1,
            height: 40,
            color: Colors.grey.shade300,
          ),
          _buildStatItem(
            Icons.air,
            _roadData!.pollutionPpm.toStringAsFixed(1),
            'ppm',
            _getPollutionColor(_roadData!.pollutionPpm),
          ),
          Container(
            width: 1,
            height: 40,
            color: Colors.grey.shade300,
          ),
          _buildStatItem(
            Icons.shield,
            '${RoadAnalysis.getSafetyScore(_roadData!)}',
            'score',
            _getScoreColor(RoadAnalysis.getSafetyScore(_roadData!)),
          ),
        ],
      ),
    );
  }

  Widget _buildCacheBanner() {
    if (!_isCachedData || _cacheTimestamp == null) {
      return const SizedBox.shrink();
    }
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.orange.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.orangeAccent),
      ),
      child: Row(
            children: [
          const Icon(Icons.info_outline, color: Colors.orange),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Offline data from ${DateFormat('MMM d, h:mm a').format(_cacheTimestamp!)}',
              style: const TextStyle(
                color: Colors.orange,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(IconData icon, String value, String unit, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 24),
              const SizedBox(height: 8),
              Text(
          value,
          style: TextStyle(
            fontSize: 20,
                  fontWeight: FontWeight.bold,
            color: color,
                ),
              ),
              Text(
          unit,
                style: TextStyle(
            fontSize: 12,
            color: Colors.grey.shade600,
                ),
              ),
            ],
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF007BFF), Color(0xFF0056D6)],
        ),
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            children: [
              Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
              Expanded(
                child: Text(
                      _roadData!.roadName,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
                      textAlign: TextAlign.center,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.share, color: Colors.white),
                onPressed: () async {
                  try {
                    final safetyScore = RoadAnalysis.getSafetyScore(_roadData!);
                    final suggestion = RoadAnalysis.getSuggestion(_roadData!);
                    final lastUpdated =
                        _roadData!.cachedAt ?? _roadData!.lastUpdated;
                    final shareText = '''
Road Eye Update

Road: ${_roadData!.roadName}
Traffic State: ${_roadData!.trafficState}
Traffic Light: ${_roadData!.trafficLightStatus}
Travel Time: ${_roadData!.travelTimeMinutes.toStringAsFixed(1)} minutes
Pollution: ${_roadData!.pollutionPpm.toStringAsFixed(2)} ppm
Safety Score: $safetyScore/100
Last Updated: ${_formatTimestamp(lastUpdated)}

Recommendation: $suggestion

${_roadData!.hasAccident ? 'Alert: Accident detected on this route.' : ''}
${_roadData!.hasEmergencyVehicle ? 'Alert: Emergency vehicle reported near this route.' : ''}

Sent from the Road Eye app
''';
                    await Share.share(shareText, subject: 'Road Information: ${_roadData!.roadName}');
                  } catch (e) {
                    if (mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Error sharing: $e')),
                      );
                    }
                  }
                },
                tooltip: 'Share',
              ),
            ],
          ),
        ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmergencyBanner() {
    List<Widget> banners = [];
    
    if (_roadData!.hasEmergencyVehicle) {
      banners.add(
        Container(
          width: double.infinity,
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.orange.shade700, Colors.orange.shade900],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
                  borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: Colors.orange.withValues(alpha: 0.6),
                blurRadius: 15,
                spreadRadius: 2,
                offset: const Offset(0, 6),
              ),
            ],
            border: Border.all(color: Colors.white.withValues(alpha: 0.3), width: 2),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.local_police, color: Colors.white, size: 40),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text(
                            'PRIORITY',
                            style: TextStyle(
                    color: Colors.white,
                              fontSize: 11,
                    fontWeight: FontWeight.bold,
                              letterSpacing: 1,
                  ),
                ),
              ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'EMERGENCY VEHICLE APPROACHING',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Clear the way immediately and expect significant delays',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.9),
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      );
    }
    
    if (_roadData!.hasAccident) {
      banners.add(
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.red.shade700, Colors.red.shade900],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(18),
                  boxShadow: [
                    BoxShadow(
                color: Colors.red.withValues(alpha: 0.5),
                blurRadius: 12,
                spreadRadius: 1,
                offset: const Offset(0, 5),
              ),
            ],
            border: Border.all(color: Colors.white.withValues(alpha: 0.2), width: 1.5),
          ),
          child: Row(
                  children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(Icons.car_crash, color: Colors.white, size: 32),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'ACCIDENT DETECTED',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _roadData!.accidentDetails ?? 'Avoid this route immediately. Emergency services are on the way.',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.9),
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
          ),
            ],
        ),
      ),
    );
  }

    if (banners.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Column(children: banners);
  }

  Widget _buildRecommendationCard() {
    Color bgColor = _getSuggestionColor(_suggestion);
    IconData icon = _getSuggestionIcon(_suggestion);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: bgColor.withValues(alpha: 0.4),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.white, size: 30),
              const SizedBox(width: 10),
              const Text(
                  'AI Route Recommendation',
                  style: TextStyle(
                    color: Colors.white,
                  fontSize: 20,
                    fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const Divider(color: Colors.white70, height: 25),
          Text(
            _suggestion,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              height: 1.5,
              fontWeight: FontWeight.w400,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSafetyScoreCard() {
    final score = RoadAnalysis.getSafetyScore(_roadData!);
    Color scoreColor = score >= 80 ? Colors.green : score >= 60 ? Colors.orange : Colors.red;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            scoreColor.withValues(alpha: 0.1),
            scoreColor.withValues(alpha: 0.05),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: scoreColor.withValues(alpha: 0.3), width: 2),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
          Column(
            children: [
              Text(
                'Safety Score',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey.shade700,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                '$score/100',
                style: TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.bold,
                  color: scoreColor,
                ),
              ),
            ],
          ),
          Container(width: 2, height: 50, color: Colors.grey.shade300),
          Column(
            children: [
              Text(
                'Route Status',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey.shade700,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                score >= 80 ? 'SAFE' : score >= 60 ? 'CAUTION' : 'RISKY',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: scoreColor,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsGrid() {
    return GridView.count(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisCount: 2,
      crossAxisSpacing: 16,
      mainAxisSpacing: 16,
      childAspectRatio: 1.1,
      children: [
        _buildMetricCard(
          icon: Icons.traffic,
          title: 'Traffic Light',
          value: _roadData!.trafficLightStatus,
          color: _getLightColor(_roadData!.trafficLightStatus),
        ),
        _buildMetricCard(
          icon: Icons.route,
          title: 'Traffic State',
          value: _roadData!.trafficState,
          color: _getTrafficColor(_roadData!.trafficState),
        ),
        _buildMetricCard(
          icon: Icons.access_time,
          title: 'Travel Time',
          value: '${_roadData!.travelTimeMinutes.toStringAsFixed(1)} min',
          color: Colors.blueGrey,
        ),
        _buildMetricCard(
          icon: Icons.air,
          title: 'Pollution',
          value: '${_roadData!.pollutionPpm.toStringAsFixed(2)} ppm',
          color: _getPollutionColor(_roadData!.pollutionPpm),
        ),
      ],
    );
  }

  Widget _buildMetricCard({
    required IconData icon,
    required String title,
    required String value,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 28),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    color: Color(0xFF64748B),
                    fontWeight: FontWeight.w500,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
              Text(
                value,
                style: TextStyle(
              fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildDirectionsButton() {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () async {
              try {
                await MapsService.openDirections(
                  _roadData!.roadName,
                  lat: _roadData!.latitude,
                  lon: _roadData!.longitude,
                );
              } catch (e) {
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('Could not open Google Maps: $e'),
                      backgroundColor: Colors.red,
                      action: SnackBarAction(
                        label: 'OK',
                        textColor: Colors.white,
                        onPressed: () {},
                      ),
                    ),
                  );
                }
              }
            },
            icon: const Icon(Icons.directions, size: 24),
            label: const Text(
              'Get Directions on Google Maps',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF007BFF),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              elevation: 4,
            ),
          ),
        ),
        const SizedBox(height: 12),
        Row(
        children: [
          Expanded(
              child: OutlinedButton.icon(
                onPressed: () {
                  Navigator.pop(context);
                },
                icon: const Icon(Icons.arrow_back),
                label: const Text('Back to Roads'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                  ),
                ),
              ],
        ),
      ],
    );
  }

  Widget _buildLastUpdatedCard() {
    final timeAgo = DateTime.now().difference(_roadData!.lastUpdated);
    String timeText = '';
    if (timeAgo.inSeconds < 60) {
      timeText = '${timeAgo.inSeconds}s ago';
    } else if (timeAgo.inMinutes < 60) {
      timeText = '${timeAgo.inMinutes}m ago';
    } else {
      timeText = '${timeAgo.inHours}h ago';
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.update, size: 16, color: Colors.grey.shade600),
          const SizedBox(width: 8),
                Text(
            'Last updated: $timeText',
                  style: TextStyle(
              fontSize: 12,
                    color: Colors.grey.shade600,
              fontStyle: FontStyle.italic,
            ),
          ),
        ],
      ),
    );
  }

  Color _getSuggestionColor(String suggestion) {
    if (suggestion.contains('EMERGENCY')) {
      return const Color(0xFFDC2626);
    } else if (suggestion.contains('CRITICAL')) {
      return const Color(0xFFDC2626);
    } else if (suggestion.contains('WARNING')) {
      return const Color(0xFFFBBF24);
    } else if (suggestion.contains('OPTIMAL')) {
      return const Color(0xFF10B981);
    } else if (suggestion.contains('CAUTION') || suggestion.contains('NOTICE')) {
      return const Color(0xFFF59E0B);
    } else {
      return const Color(0xFF007BFF);
    }
  }

  IconData _getSuggestionIcon(String suggestion) {
    if (suggestion.contains('EMERGENCY')) {
      return Icons.warning_amber_rounded;
    } else if (suggestion.contains('CRITICAL')) {
      return Icons.error_outline;
    } else if (suggestion.contains('WARNING')) {
      return Icons.warning_rounded;
    } else if (suggestion.contains('OPTIMAL')) {
      return Icons.check_circle_outline;
    } else {
      return Icons.info_outline;
    }
  }

  Color _getLightColor(String status) {
    switch (status) {
      case 'Green': return Colors.green;
      case 'Yellow': return Colors.amber;
      case 'Red': return Colors.red;
      default: return Colors.grey;
    }
  }

  Color _getTrafficColor(String state) {
    switch (state) {
      case 'Heavy Congestion': return Colors.red.shade700;
      case 'Moderate': return Colors.amber.shade700;
      case 'Clear': return Colors.green.shade700;
      default: return Colors.blueGrey;
    }
  }

  Color _getPollutionColor(double ppm) {
    if (ppm > 15.0) return Colors.purple.shade700;
    if (ppm > 10.0) return Colors.orange.shade700;
      return Colors.lightBlue.shade700;
    }

  Color _getScoreColor(int score) {
    if (score >= 80) return Colors.green;
    if (score >= 60) return Colors.orange;
    return Colors.red;
  }
}

// Settings Page

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  bool _notificationsEnabled = true;
  bool _autoRefreshEnabled = true;
  int _refreshInterval = 3;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF007BFF), Color(0xFF0056D6)],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back, color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const Expanded(
                      child: Text(
                        'Settings',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    const SizedBox(width: 48),
                  ],
                ),
              ),
              Expanded(
                child: Container(
                  decoration: const BoxDecoration(
                    color: Color(0xFFF5F7FA),
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(30),
                      topRight: Radius.circular(30),
                    ),
                  ),
                  child: ListView(
                    padding: const EdgeInsets.all(20),
                    children: [
                      _buildSettingsSection(
                        'Notifications',
                        [
                          _buildSwitchTile(
                            'Enable Notifications',
                            'Get alerts for emergencies and traffic updates',
                            _notificationsEnabled,
                            (value) => setState(() => _notificationsEnabled = value),
                            Icons.notifications_active,
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      _buildSettingsSection(
                        'Data & Refresh',
                        [
                          _buildSwitchTile(
                            'Auto Refresh',
                            'Automatically update road data',
                            _autoRefreshEnabled,
                            (value) => setState(() => _autoRefreshEnabled = value),
                            Icons.refresh,
                          ),
                          _buildListTile(
                            'Refresh Interval',
                            '$_refreshInterval seconds',
                            Icons.timer,
                            () {
                              showDialog(
                                context: context,
                                builder: (context) => AlertDialog(
                                  title: const Text('Refresh Interval'),
                                  content: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [3, 5, 10, 15, 30]
                                        .map((seconds) => ListTile(
                                              title: Text('$seconds seconds'),
                                              onTap: () {
                                                setState(() => _refreshInterval = seconds);
                                                Navigator.pop(context);
                                              },
                                            ))
                                        .toList(),
                                  ),
                                ),
                              );
                            },
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      _buildSettingsSection(
                        'About',
                        [
                          _buildListTile(
                            'App Version',
                            '1.0.0',
                            Icons.info,
                            null,
                          ),
                          _buildListTile(
                            'Firebase Status',
                            'Connected',
                            Icons.cloud_done,
                            null,
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      _buildSettingsSection(
                        'Support',
                        [
                          _buildListTile(
                            'Help & FAQ',
                            'Get help using the app',
                            Icons.help_outline,
                            () {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(content: Text('Help section coming soon!')),
                              );
                            },
                          ),
                          _buildListTile(
                            'Report Issue',
                            'Report a problem',
                            Icons.bug_report,
                            () {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(content: Text('Report issue feature coming soon!')),
                              );
                            },
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingsSection(String title, List<Widget> children) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 12),
          child: Text(
            title,
          style: TextStyle(
              fontSize: 16,
            fontWeight: FontWeight.bold,
              color: Colors.grey.shade700,
          ),
        ),
        ),
        Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(children: children),
        ),
      ],
    );
  }

  Widget _buildSwitchTile(
    String title,
    String subtitle,
    bool value,
    ValueChanged<bool> onChanged,
    IconData icon,
  ) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: const Color(0xFF007BFF).withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(icon, color: const Color(0xFF007BFF), size: 24),
      ),
      title: Text(
        title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
          fontSize: 16,
        ),
      ),
      subtitle: Text(
        subtitle,
        style: TextStyle(
          fontSize: 13,
          color: Colors.grey.shade600,
        ),
      ),
      trailing: Switch(
        value: value,
        onChanged: onChanged,
        activeThumbColor: const Color(0xFF007BFF),
      ),
    );
  }

  Widget _buildListTile(
    String title,
    String subtitle,
    IconData icon,
    VoidCallback? onTap,
  ) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
          color: const Color(0xFF007BFF).withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(icon, color: const Color(0xFF007BFF), size: 24),
      ),
      title: Text(
        title,
        style: const TextStyle(
          fontWeight: FontWeight.w600,
          fontSize: 16,
        ),
      ),
      subtitle: Text(
        subtitle,
        style: TextStyle(
          fontSize: 13,
          color: Colors.grey.shade600,
        ),
      ),
      trailing: onTap != null
          ? const Icon(Icons.chevron_right, color: Colors.grey)
          : null,
      onTap: onTap,
    );
  }
}

void _handleNotificationTap(String? payload) {
  if (payload == null) return;
  try {
    final Map<String, dynamic> data = jsonDecode(payload);
    final roadId = data['roadId']?.toString();
    _handleNotificationNavigation(roadId);
  } catch (e) {
    debugPrint('Failed to handle notification payload: $e');
  }
}

void _handleNotificationNavigation(String? roadId) {
  if (roadId == null) return;
  navigatorKey.currentState?.push(
    MaterialPageRoute(
      builder: (_) => RoadDetailsPage(roadId: roadId),
    ),
  );
}