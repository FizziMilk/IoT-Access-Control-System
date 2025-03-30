import React, { useState, useCallback } from 'react';
import { View, Text, TextInput, Button,Switch, FlatList, Alert, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { backendIP } from "../../../config";
import SchedulePicker from "../../../components/SchedulePicker";

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
    const [loading, setLoading] = useState(true);
    const [selectedUser, setSelectedUser] = useState<User | null>(null);
    const [showSchedulePicker, setShowSchedulePicker] = useState(false);

    const fetchUsers = async () => {
        try {
            const response = await fetch(`${backendIP}/users`);
            if (!response.ok) {
                throw new Error("Failed to fetch users");
            }
            const data = await response.json();
            setUsers(data);
        } catch (error) {
            Alert.alert("Error", "Failed to load users");
        } finally {
            setLoading(false);
        }
    };
    
    // Fetch users when the screen comes into focus
    useFocusEffect(
        React.useCallback(() => {
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

    const toggleUserAccess = async (userId: number, currentStatus: boolean) => {
        try {
            const response = await fetch(`${backendIP}/users`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    id: userId,
                    is_allowed: !currentStatus,
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to update user access");
            }

            // Update local state
            setUsers((prevUsers) =>
                prevUsers.map((user) =>
                    user.id === userId
                        ? { ...user, is_allowed: !currentStatus }
                        : user
                )
            );

            Alert.alert("Success", "User access updated successfully");
        } catch (error) {
            Alert.alert("Error", "Failed to update user access");
        }
    };

    const handleScheduleCreated = () => {
        setShowSchedulePicker(false);
        fetchUsers(); // Refresh the user list
    };

    if (loading) {
        return (
            <View style={styles.center}>
                <ActivityIndicator size="large" color="#0000ff" />
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Add User</Text>
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
                keyExtractor={(item) => item.id.toString()}
                renderItem={({ item }) => (
                    <View style={styles.userItem}>
                        <View style={styles.userInfo}>
                            <Text style={styles.userName}>
                                {item.name || "Unnamed User"}
                            </Text>
                            <Text style={styles.phoneNumber}>{item.phone_number}</Text>
                        </View>
                        <View style={styles.actions}>
                            <TouchableOpacity
                                style={[
                                    styles.accessButton,
                                    item.is_allowed ? styles.allowed : styles.notAllowed,
                                ]}
                                onPress={() => toggleUserAccess(item.id, item.is_allowed)}
                            >
                                <Text style={styles.buttonText}>
                                    {item.is_allowed ? "Revoke Access" : "Grant Access"}
                                </Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                                style={styles.scheduleButton}
                                onPress={() => {
                                    setSelectedUser(item);
                                    setShowSchedulePicker(true);
                                }}
                            >
                                <Text style={styles.buttonText}>Set Schedule</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                )}
            />
            {showSchedulePicker && selectedUser && (
                <SchedulePicker
                    userId={selectedUser.id}
                    onScheduleCreated={handleScheduleCreated}
                />
            )}
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
    userItem: {
        backgroundColor: "#f9f9f9",
        padding: 16,
        borderRadius: 8,
        marginBottom: 12,
    },
    userInfo: {
        marginBottom: 8,
    },
    userName: {
        fontSize: 16,
        fontWeight: "bold",
    },
    phoneNumber: {
        fontSize: 14,
        color: "#666",
    },
    actions: {
        flexDirection: "row",
        justifyContent: "space-between",
        gap: 8,
    },
    accessButton: {
        flex: 1,
        padding: 8,
        borderRadius: 4,
        alignItems: "center",
    },
    allowed: {
        backgroundColor: "#ff3b30",
    },
    notAllowed: {
        backgroundColor: "#34c759",
    },
    scheduleButton: {
        flex: 1,
        padding: 8,
        backgroundColor: "#007AFF",
        borderRadius: 4,
        alignItems: "center",
    },
    buttonText: {
        color: "#fff",
        fontSize: 14,
        fontWeight: "bold",
    },
    center: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
    },
});