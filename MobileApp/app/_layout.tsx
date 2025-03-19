import { AuthProvider } from '@/providers/auth';
import { Stack } from 'expo-router';
import React from 'react';

export default function RootLayout() {
  return (
    <AuthProvider>
      <Stack>
        <Stack.Screen name="(authorized)" options={{ headerShown: false }} />
        <Stack.Screen name="/login" options={{ headerShown: false }} />
      </Stack>
    </AuthProvider>
  );
}