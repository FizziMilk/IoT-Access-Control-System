import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button,Switch, FlatList, Alert, StyleSheet } from 'react-native';

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;

interface User {
    id: number;
    name: string | null;
    phone_number: string;
    is_allowed: boolean;
}

export default function Management() {
    const [phoneNumber, setPhoneNumber ] = useState('');
    const [users, setUsers] = useState<User[]>([]);

    // Fetch users from the backend when the component mounts
    useEffect(() => {
        fetch(`${backendIP}/users`)
        .then(res => res.json())
        .then((data:User[]) => setUsers(data))
        .catch(err => {
            console.error("Error fetching users:", err);
            Alert.alert("Error", "Failed to fetch users");    
        });
    }, []);

    // Update a user's permission via the backend
    const updateUserPermission = (userId: number, newPermission: boolean) => {
        fetch(`${backendIP}/users`,{
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({id: userId, is_allowed: newPermission}),
        })
            .then(res => res.json())
            .then(data => {
                Alert.alert("Success", "User updated successfully");
                setUsers(prev =>
                    prev.map(user =>
                        user.id == userId ? { ...user, is_allowed: newPermission } : user
                    )
                );
            })
            .catch(err => {
                console.error("Error updating user:", err);
                Alert.alert("Error", "Failed to update user");
            });
    };

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

            {/* User Management Section */}
            <Text style={[styles.title, {fontSize: 20, marginTop: 32}]}> User Management</Text>
            <FlatList
                data={users}
                keyExtractor= {(item) => item.id.toString()}
                renderItem = {({ item }) => (
                    <View style = {styles.userRow}>
                        <Text style = {styles.userText}>{item.name ? item.name : item.phone_number}</Text>
                        <Switch
                            value={item.is_allowed}
                            onValueChange={(value) => updateUserPermission(item.id,value)}
                    />
                    <Text style ={styles.permissionText}>
                        {item.is_allowed ? "Allowed" : "Not Allowed"}
                    </Text>
                </View>
            )}
        />
    </View>
);
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
    userRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: 8,
    },
    userText: {
        flex: 1,
        fontSize: 16,
    },
    permissionText: {
        marginLeft: 8,
        fontSize: 16,
    },
});