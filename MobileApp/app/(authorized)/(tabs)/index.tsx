import { StyleSheet, View, Button, Pressable, Text } from 'react-native';
import { useRouter } from 'expo-router';
import ImageViewer from '@/components/ImageViewer';
import { useAuth } from '@/providers/auth';

const PlaceholderImage = require("@/assets/images/background-image.png");
const backendIP = process.env.BACKEND_IP;

export default function Index() {
  const { signOut } = useAuth();
  const router = useRouter();

  const handleSignOut = () => {
    signOut();
    router.replace('/login');
  };

  //Sends a POST to the webserver to unlock the door
  const unlockDoor = async () => {
    fetch(`${backendIP}/unlock`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'
      },
      body: JSON.stringify({ command: 'open_door'})
  })
    .then(response => response.json())
    .then(data => {
      console.log('Unlock response:', data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
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