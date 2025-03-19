import React, { useState } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet } from 'react-native';
import {useRouter } from 'expo-router';
import { useAuth } from '../../providers/auth';

export default function LoginScreen() {
    const { signIn } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [otp, setOtp] = useState('');
    const [stage, setStage] = useState<'login' | 'otp'>('login');
    const router = useRouter();
    //Secret backend ip stored in .env
    const backendIP = process.env.BACKEND_IP;

    //Sends a POST to the webserver with the user's login information and waits for response
    //Replies with OTP prompt if details correct
    const handleLogin = async () => {
        try {
            const response = await fetch(`${backendIP}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json'},
                body: JSON.stringify({ username, password}),
            });
             const data = await response.json();
            if (response.ok) {
            Alert.alert('OTP Sent', 'Please check your phone.');
            setStage('otp');
             } else {
                Alert.alert('Login Failed', data.error || 'Unknown error');
             }
        } catch (error) {
            const err = error as Error;
            Alert.alert('Error',err.message);
        }
    };
    //Sends a POST to the webserver with the OTP and the username of the user sending it
    //Waits for ok and token, calls signIn with the token and routes to correct page
    const handleVerifyOTP = async () => {
        try {
            const response = await fetch (`${backendIP}/verify-otp`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json'},
                body: JSON.stringify({ username, otp}),
            });
            const data = await response.json();
            if (response.ok && data.status === "approved" && data.token) {
                await signIn(data.token);
                router.replace('../(authorized)/(tabs)');
            } else {
                Alert.alert('OTP Verification Failed', data.error || 'Unknown error');
            }
            } catch (error) {
                const err = error as Error;
                Alert.alert('Error', err.message);
            }
        };
    //Page layout, login and OTP boxes
    return (
        <View style = {styles.container}>
            {stage === 'login' ? (
                <>
                <Text style = {styles.title}>Admin Login</Text>
                <TextInput
                    style ={styles.input}
                    placeholder= "Username"
                    value={username}
                    onChangeText={setUsername}
                    autoCapitalize="none"
                />
                <TextInput
                    style={styles.input}
                    placeholder= "Password"
                    value={password}
                    onChangeText={setPassword}
                    secureTextEntry
                />
                <Button title="Login" onPress={handleLogin} />
                </>
            ) : (
             <>
                <Text style={styles.title}>Enter OTP</Text>
                <TextInput 
                    style={styles.input}
                    placeholder = "OTP Code"
                    value = {otp}
                    onChangeText = {setOtp}
                    keyboardType = "numeric"
                />
                <Button title= "Verify OTP" onPress={handleVerifyOTP} />
            </>
            )}
        </View>
    );   
}
    // Page Styling
    const styles = StyleSheet.create({
        container: {flex: 1, padding: 16, justifyContent: 'center'},
        title: { fontSize: 24, textAlign: 'center', marginBottom: 24},
        input: { 
            height: 50,
            borderColor: '#25292e',
            borderWidth:1,
            paddingHorizontal: 8,
            marginBottom: 16,
            borderRadius:4,
        }
    });
