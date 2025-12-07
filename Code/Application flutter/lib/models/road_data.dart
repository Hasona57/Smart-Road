class RoadData {
  final String roadName;
  final String roadId;
  final String trafficLightStatus;
  final String trafficState;
  final double pollutionPpm;
  final double travelTimeMinutes;
  final bool hasAccident;
  final bool hasEmergencyVehicle;
  final String? accidentDetails;
  final DateTime lastUpdated;
  final double? latitude;
  final double? longitude;
  final bool isFromCache;
  final DateTime? cachedAt;

  const RoadData({
    required this.roadName,
    required this.roadId,
    required this.trafficLightStatus,
    required this.trafficState,
    required this.pollutionPpm,
    required this.travelTimeMinutes,
    this.hasAccident = false,
    this.hasEmergencyVehicle = false,
    this.accidentDetails,
    required this.lastUpdated,
    this.latitude,
    this.longitude,
    this.isFromCache = false,
    this.cachedAt,
  });

  factory RoadData.fromRealtimeJson(Map<String, dynamic> json) {
    return RoadData(
      roadName: json['roadName'] ?? 'Unknown Road',
      roadId: json['roadId']?.toString() ?? '',
      trafficLightStatus: json['trafficLightStatus'] ?? 'Unknown',
      trafficState: json['trafficState'] ?? 'Unknown',
      pollutionPpm: (json['pollutionPpm'] ?? 0.0).toDouble(),
      travelTimeMinutes: (json['travelTimeMinutes'] ?? 0.0).toDouble(),
      hasAccident: json['hasAccident'] ?? false,
      hasEmergencyVehicle: json['hasEmergencyVehicle'] ?? false,
      accidentDetails: json['accidentDetails'],
      lastUpdated: json['lastUpdated'] != null
          ? DateTime.tryParse(json['lastUpdated'].toString()) ?? DateTime.now()
          : DateTime.now(),
      latitude: json['latitude'] != null ? (json['latitude'] as num).toDouble() : null,
      longitude: json['longitude'] != null ? (json['longitude'] as num).toDouble() : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'roadName': roadName,
      'roadId': roadId,
      'trafficLightStatus': trafficLightStatus,
      'trafficState': trafficState,
      'pollutionPpm': pollutionPpm,
      'travelTimeMinutes': travelTimeMinutes,
      'hasAccident': hasAccident,
      'hasEmergencyVehicle': hasEmergencyVehicle,
      'accidentDetails': accidentDetails,
      'lastUpdated': lastUpdated.toIso8601String(),
      'latitude': latitude,
      'longitude': longitude,
    };
  }

  RoadData copyWith({
    String? roadName,
    String? roadId,
    String? trafficLightStatus,
    String? trafficState,
    double? pollutionPpm,
    double? travelTimeMinutes,
    bool? hasAccident,
    bool? hasEmergencyVehicle,
    String? accidentDetails,
    DateTime? lastUpdated,
    double? latitude,
    double? longitude,
    bool? isFromCache,
    DateTime? cachedAt,
  }) {
    return RoadData(
      roadName: roadName ?? this.roadName,
      roadId: roadId ?? this.roadId,
      trafficLightStatus: trafficLightStatus ?? this.trafficLightStatus,
      trafficState: trafficState ?? this.trafficState,
      pollutionPpm: pollutionPpm ?? this.pollutionPpm,
      travelTimeMinutes: travelTimeMinutes ?? this.travelTimeMinutes,
      hasAccident: hasAccident ?? this.hasAccident,
      hasEmergencyVehicle: hasEmergencyVehicle ?? this.hasEmergencyVehicle,
      accidentDetails: accidentDetails ?? this.accidentDetails,
      lastUpdated: lastUpdated ?? this.lastUpdated,
      latitude: latitude ?? this.latitude,
      longitude: longitude ?? this.longitude,
      isFromCache: isFromCache ?? this.isFromCache,
      cachedAt: cachedAt ?? this.cachedAt,
    );
  }
}


