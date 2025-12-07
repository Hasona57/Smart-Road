import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/road_data.dart';

class CachedRoadBundle {
  final Map<String, RoadData> roads;
  final DateTime timestamp;

  const CachedRoadBundle({
    required this.roads,
    required this.timestamp,
  });
}

class RoadCacheManager {
  static const String _roadsKey = 'cached_roads';
  static const String _timestampKey = 'cached_roads_timestamp';

  Future<void> saveRoads(Map<String, RoadData> roads) async {
    final prefs = await SharedPreferences.getInstance();
    final Map<String, dynamic> serialized = {};
    for (final entry in roads.entries) {
      serialized[entry.key] = entry.value.toJson();
    }
    await prefs.setString(_roadsKey, jsonEncode(serialized));
    await prefs.setString(_timestampKey, DateTime.now().toIso8601String());
  }

  Future<void> saveRoad(RoadData road) async {
    final prefs = await SharedPreferences.getInstance();
    final existing = prefs.getString(_roadsKey);
    final Map<String, dynamic> data =
        existing != null ? jsonDecode(existing) as Map<String, dynamic> : {};
    data[road.roadId] = road.toJson();
    await prefs.setString(_roadsKey, jsonEncode(data));
    await prefs.setString(_timestampKey, DateTime.now().toIso8601String());
  }

  Future<CachedRoadBundle?> loadRoads() async {
    final prefs = await SharedPreferences.getInstance();
    final cached = prefs.getString(_roadsKey);
    final timestampString = prefs.getString(_timestampKey);
    if (cached == null || cached.isEmpty || timestampString == null) {
      return null;
    }
    final DateTime timestamp = DateTime.tryParse(timestampString) ?? DateTime.now();
    final Map<String, dynamic> decoded = jsonDecode(cached) as Map<String, dynamic>;

    final Map<String, RoadData> roads = {};
    decoded.forEach((key, value) {
      final roadData = RoadData.fromRealtimeJson(Map<String, dynamic>.from(value))
          .copyWith(isFromCache: true, cachedAt: timestamp);
      roads[key] = roadData;
    });

    return CachedRoadBundle(roads: roads, timestamp: timestamp);
  }

  Future<RoadData?> loadRoad(String roadId) async {
    final bundle = await loadRoads();
    if (bundle == null) return null;
    return bundle.roads[roadId];
  }

  Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_roadsKey);
    await prefs.remove(_timestampKey);
  }
}

