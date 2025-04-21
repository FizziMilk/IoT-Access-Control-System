import { StyleSheet, View, Button, Pressable, Text, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import ImageViewer from '@/components/ImageViewer';
import { useAuth } from '@/providers/auth';
import * as SecureStore from 'expo-secure-store';

const PlaceholderImage = require("@/assets/images/background-image.png");
const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;

export default function Index() {
  const { signOut } = useAuth();
  const router = useRouter();

  const handleSignOut = () => {
    signOut();
    router.replace('/login');
  };

  //Sends a POST to the webserver to unlock the door
  const unlockDoor = async () => {
    try {
      // Get the JWT token from secure storage
      const token = await SecureStore.getItemAsync('accessToken');
      console.log('[DEBUG] Retrieved token from secure storage, length:', token ? token.length : 'null');
      
      if (!token) {
        Alert.alert('Error', 'Not authenticated. Please log in again.');
        return;
      }
      
      console.log('[DEBUG] Sending request to:', `${backendIP}/unlock`);
      console.log('[DEBUG] Using Authorization header:', `Bearer ${token.substring(0, 10)}...`);
      
      // Send the request with the auth token
      const response = await fetch(`${backendIP}/unlock`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ command: 'unlock_door'})
      });
      
      console.log('[DEBUG] Response status:', response.status);
      
      const data = await response.json();
      console.log('[DEBUG] Unlock response:', data);
      
      // Show success message
      Alert.alert('Success', 'Door unlock command sent');
    } catch (error) {
      console.error('[ERROR] Door unlock error:', error);
      Alert.alert('Error', 'Failed to send unlock command');
    }
  }

  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <ImageViewer imgSource={PlaceholderImage} />
      </View>

      <View
        style={[
          styles.buttonContainer,
          { borderWidth: 4, borderColor: '#ffd33d', borderRadius: 18, marginTop: 10 }, // Adjust marginTop to move the button higher
        ]}
      >
        <Pressable style={styles.button} onPress={unlockDoor}>
          <Text style={styles.buttonLabel}>Open Door</Text>
        </Pressable>
      </View>

      <View style={styles.footerContainer}>
        <Button title="Sign Out" onPress={handleSignOut} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#25292e',
    alignItems: 'center',
    justifyContent: 'flex-start', // Align items to the top
    paddingTop: 20, // Add padding to move content down
  },
  imageContainer: {
    flex: 1,
    paddingTop: 28,
  },
  footerContainer: {
    flex: 1 / 3,
    alignItems: 'center',
  },
  button: {
    fontSize: 20,
    textDecorationLine: 'underline',
    color: '#fff',
    marginBottom: 0,
  },
  buttonContainer: {
    width: 320,
    height: 68,
    marginHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 2,
  },
  buttonLabel: {
    color: '#fff',
    fontSize: 18,
  },
});