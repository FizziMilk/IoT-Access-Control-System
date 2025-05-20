import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import {useRouter } from 'expo-router';
import { useAuth } from '../../providers/auth';

export default function LoginScreen() {
    const { signIn } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [otp, setOtp] = useState('');
    const [stage, setStage] = useState<'login' | 'otp'>('login');
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();
    const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;
    
    useEffect(() => {
        console.log('[Login] Backend IP configured as:', backendIP);
        // Test connection to backend
        testConnection();
    }, []);

    const testConnection = async () => {
        try {
            const url = `${backendIP}/login`;
            console.log('[Login] Testing connection to:', url);
            const response = await fetch(url, {
                method: 'OPTIONS',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                }
            });
            console.log('[Login] Backend connection test successful:', response.status);
        } catch (error) {
            console.error('[Login] Backend connection test failed:', error);
        }
    };

    //Sends a POST to the webserver with the user's login information and waits for response
    //Replies with OTP prompt if details correct
    const handleLogin = async () => {
        if (!username || !password) {
            Alert.alert('Error', 'Please enter both username and password');
            return;
        }

        setIsLoading(true);
        try {
            const url = `${backendIP}/login`;
            console.log('[Login] Attempting login...');
            console.log('[Login] URL:', url);
            
            const response = await fetch(url, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ username, password }),
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('[Login] Error response:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('[Login] Response data:', JSON.stringify(data, null, 2));
            
            if (response.ok) {
                Alert.alert('Success', 'OTP Sent. Please check your phone.');
                setStage('otp');
            } else {
                Alert.alert('Login Failed', data.error || 'Invalid credentials');
            }
        } catch (error: any) {
            console.error('[Login] Error details:', {
                message: error.message,
                stack: error.stack,
                name: error.name
            });
            Alert.alert(
                'Connection Error',
                'Could not connect to the server. Please check your internet connection and try again.'
            );
        } finally {
            setIsLoading(false);
        }
    };
    //Sends a POST to the webserver with the OTP and the username of the user sending it
    //Waits for ok and token, calls signIn with the token and routes to correct page
    const handleVerifyOTP = async () => {
        if (!otp) {
            Alert.alert('Error', 'Please enter the OTP code');
            return;
        }

        setIsLoading(true);
        try {
            const url = `${backendIP}/verify-otp`;
            console.log('[Login] Attempting OTP verification...');
            console.log('[Login] URL:', url);
            
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json'},
                body: JSON.stringify({ username, otp }),
            });
            
            const data = await response.json();
            console.log('[Login] OTP verification response:', JSON.stringify(data, null, 2));
            
            if (response.ok && data.status === "approved" && data.token) {
                await signIn(data.token);
                router.replace('../(authorized)/(tabs)');
            } else {
                Alert.alert('Verification Failed', data.error || 'Invalid OTP code');
            }
        } catch (error: any) {
            console.error('[Login] OTP verification error:', {
                message: error.message,
                stack: error.stack,
                name: error.name
            });
            Alert.alert(
                'Connection Error',
                'Could not verify OTP. Please check your connection and try again.'
            );
        } finally {
            setIsLoading(false);
        }
    };
    //Page layout, login and OTP boxes
    return (
        <View style={styles.container}>
            {stage === 'login' ? (
                <>
                    <Text style={styles.title}>Admin Login</Text>
                    <TextInput
                        style={styles.input}
                        placeholder="Username"
                        value={username}
                        onChangeText={setUsername}
                        autoCapitalize="none"
                        editable={!isLoading}
                    />
                    <TextInput
                        style={styles.input}
                        placeholder="Password"
                        value={password}
                        onChangeText={setPassword}
                        secureTextEntry
                        editable={!isLoading}
                    />
                    {isLoading ? (
                        <ActivityIndicator size="large" color="#0000ff" />
                    ) : (
                        <Button title="Login" onPress={handleLogin} />
                    )}
                </>
            ) : (
                <>
                    <Text style={styles.title}>Enter OTP</Text>
                    <TextInput 
                        style={styles.input}
                        placeholder="OTP Code"
                        value={otp}
                        onChangeText={setOtp}
                        keyboardType="numeric"
                        editable={!isLoading}
                    />
                    {isLoading ? (
                        <ActivityIndicator size="large" color="#0000ff" />
                    ) : (
                        <>
                            <Button title="Verify OTP" onPress={handleVerifyOTP} />
                            <Button title="Back to Login" onPress={() => setStage('login')} color="#888" />
                        </>
                    )}
                </>
            )}
        </View>
    );   
}
    // Page Styling
    const styles = StyleSheet.create({
        container: {
            flex: 1,
            padding: 16,
            justifyContent: 'center',
            backgroundColor: '#fff'
        },
        title: {
            fontSize: 24,
            textAlign: 'center',
            marginBottom: 24
        },
        input: { 
            height: 50,
            borderColor: '#25292e',
            borderWidth: 1,
            paddingHorizontal: 8,
            marginBottom: 16,
            borderRadius: 4,
        }
    });
