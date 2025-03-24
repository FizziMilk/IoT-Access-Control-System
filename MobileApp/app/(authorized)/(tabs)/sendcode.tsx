import React, { useState } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;

export default function SendCodeScreen() {
    const [phoneNumber, setPhoneNumber ] = useState('');
    const [scheduleTime, setScheduleTime] = useState <Date | null>(null);
    const [showPicker, setShowPicker] = useState(false);

const handleSendOTP = async () => {
    try {
        const response = await fetch(`${backendIP}/start-verification`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phone_number: phoneNumber}),
        });
        const data = await response.json();
        if (response.ok){
            Alert.alert('OTP Sent', 'OTP has been sent to the specified phone number.');
        } else {
         Alert.alert('Error', data.error || 'Unknown error');
        }
    } catch (error) {
        const err = error as Error;
        Alert.alert('Error', err.message);
    }
};
return (
    <View style = {styles.container}>
        <Text style = {styles.title}>Send OTP</Text>
        <TextInput
            style={styles.input}
            placeholder="Phone Number"
            value={phoneNumber}
            onChangeText={setPhoneNumber}
            keyboardType="phone-pad"
        />
        <Button title= "Send OTP Now" onPress={handleSendOTP}/>
        </View>
)
}

const styles = StyleSheet.create({
    container: { flex: 1, padding: 16, justifyContent: 'center'},
    title: { fontSize: 24, textAlign: 'center', marginBottom: 24},
    input: {
        height: 50,
        borderColor: '#25292e',
        borderWidth: 1,
        paddingHorizontal: 8,
        marginBottom:16,
        borderRadius: 4,
    },
});