import React from 'react';
import { View, StyleSheet, Button, Text } from 'react-native';
import { Link, Stack, useRouter } from 'expo-router';
import { useAuth } from '@/providers/auth';

export default function NotFoundScreen() {
  const { signOut } = useAuth();
  const router = useRouter();

  const handleSignOut = () => {
    signOut();
    router.replace('/login');
  };

  return (
    <>
      <Stack.Screen options={{ title: 'Oops! Not Found' }} />
      <View style={styles.container}>
        <Text style = {styles.title}> Page not found!</Text>
        <Link href="./" style={styles.button}>
          Go back to Home screen!
        </Link>
        <Button title="Sign Out" onPress={handleSignOut} />
      </View>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#25292e',
    justifyContent: 'center',
    alignItems: 'center',
  },
  button: {
    fontSize: 20,
    textDecorationLine: 'underline',
    color: '#fff',
    marginBottom: 20,
  },
  title: { fontSize: 24, textAlign: 'center', marginBottom: 24, color: '#fff' },
});