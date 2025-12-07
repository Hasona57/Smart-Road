# Custom Notification Sounds

To add custom notification sounds that say "accident" or "emergency":

1. Create or obtain audio files (MP3 format recommended)
   - `accident.mp3` - Should say "accident" or have a distinct accident alert sound
   - `emergency.mp3` - Should say "emergency" or have a distinct emergency vehicle sound

2. Place the files in this directory: `android/app/src/main/res/raw/`

3. Update `lib/services/notification_service.dart`:
   - Find the accident channel creation (around line 81)
   - Change: `sound: const RawResourceAndroidNotificationSound('accident')`
   - Find the emergency channel creation (around line 95)
   - Change: `sound: const RawResourceAndroidNotificationSound('emergency')`

4. Rebuild the app: `flutter build apk`

## Creating Audio Files

You can create audio files using:
- Text-to-Speech (TTS) tools online
- Voice recording apps
- Audio editing software like Audacity

Make sure the files are:
- Short (1-3 seconds recommended)
- Clear and recognizable
- In MP3 or WAV format
- Named exactly: `accident.mp3` and `emergency.mp3`

