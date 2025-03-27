import React, { useState, useCallback } from 'react';
import { View, Text, TextInput, Button,Switch, FlatList, Alert, StyleSheet, TouchableOpacity } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;

interface User {
    id: number;
    name: string | null;
    phone_number: string;
    is_allowed: boolean;
}

export default function Management() {
    const [phoneNumber, setPhoneNumber ] = useState('');
    const [name, setName] = useState('');
    const [users, setUsers] = useState<User[]>([]);

    const fetchUsers = () => {
        fetch(`${backendIP}/users`)
          .then(res => res.json())
          .then((data: User[]) => setUsers(data))
          .catch(err => {
            console.error("Error fetching users:", err);
            Alert.alert("Error", "Failed to fetch users");
          });
      };
    
      // Fetch users when the screen comes into focus
      useFocusEffect(
        useCallback(() => {
          fetchUsers();
        }, [])
      );

    
      const addUser = async () => {
        if (!phoneNumber) {
            Alert.alert("Error", "Phone number is required");
            return;
        }
        try{
            const response = await fetch(`${backendIP}/users`,{
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name,phone_number: phoneNumber}),
            });
            const data = await response.json();
            if (response.ok) {
                Alert.alert("Success", "User added successfully");
                setUsers(prev => [...prev, { id: data.id, name, phone_number: phoneNumber, is_allowed: false}]);
                setName('');
                setPhoneNumber('');
            } else {
                Alert.alert("Error", data.error || "Failed to add user");
            }
        } catch (error) {
            const err = error as Error;
            Alert.alert("Error", err.message);
        }
    };

        const removeUser = async (userId: number) => {
            try {
                const response = await fetch(`${backendIP}/users`,{
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({id: userId}),
                });
                const data = await response.json();
                if (response.ok) {
                    Alert.alert("Success", "User removed successfully");
                    setUsers(prev => prev.filter(user => user.id !== userId));
                } else {
                    Alert.alert("Error",data.error || "Failed to remove user");
                }
            } catch(error){
                const err = error as Error;
                Alert.alert("Error", err.message);
            }
        }

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

    /* Legacy code to send a user an OTP, not in use currently
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
    }; */
    return (
        <View style = {styles.container}>
            <Text style = {styles.title}>Add User</Text>
            <TextInput
                style={styles.input}
                placeholder="Name (optional)"
                value={name}
                onChangeText={setName}
            />
            <TextInput
                style={styles.input}
                placeholder="Phone Number (With country code)"
                value={phoneNumber}
                onChangeText={setPhoneNumber}
                keyboardType="phone-pad"
            />
            <Button title= "Add User" onPress={addUser}/>

            {/* User Management Section */}
            <Text style={[styles.title, {fontSize: 20, marginTop: 32}]}> User Management</Text>
            <FlatList
                data={users}
                keyExtractor= {(item) => item.id.toString()}
                renderItem = {({ item }) => (
                    <View style = {styles.userRow}>
                        
                    <TouchableOpacity
                        style={styles.removeButton}
                        onPress={() => removeUser(item.id)}
                    >
                        <Text style = {styles.removeButtonText}>Remove</Text>
                    </TouchableOpacity>
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
        justifyContent: 'space-between',
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
    removeButton: {
        backgroundColor: 'red',
        paddingVertical: 4,
        paddingHorizontal: 8,
        borderRadius: 8,
        marginRight: 8,
    },
    removeButtonText: {
        color: 'white',
        fontSize: 12,
        textAlign: 'center',
    },
});