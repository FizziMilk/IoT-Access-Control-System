import React, { useState, useCallback, useRef, useEffect } from 'react';
import { View, Text, TextInput, Button, Switch, FlatList, Alert, StyleSheet, TouchableOpacity, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView, Keyboard, findNodeHandle } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { backendIP } from "../../../config";
import SchedulePicker from "../../../components/SchedulePicker";
import { Ionicons } from '@expo/vector-icons';

interface User {
    id: number;
    name: string | null;
    phone_number: string;
    is_allowed: boolean;
    low_security: boolean;
}

export default function Management() {
    const [phoneNumber, setPhoneNumber ] = useState('');
    const [name, setName] = useState('');
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedUser, setSelectedUser] = useState<User | null>(null);
    const [showSchedulePicker, setShowSchedulePicker] = useState(false);
    const [editingUser, setEditingUser] = useState<number | null>(null);
    const [editName, setEditName] = useState('');
    const flatListRef = useRef<FlatList>(null);
    const scrollViewRef = useRef<ScrollView>(null);
    const itemRefs = useRef<{[key: string]: any}>({});
    const [keyboardVisible, setKeyboardVisible] = useState(false);

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
                setUsers(prev => [...prev, { id: data.id, name, phone_number: phoneNumber, is_allowed: false, low_security: false}]);
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

    const toggleLowSecurity = async (userId: number, currentStatus: boolean) => {
        try {
            // Find the current user to get their is_allowed status
            const currentUser = users.find(user => user.id === userId);
            if (!currentUser) {
                throw new Error("User not found");
            }

            const response = await fetch(`${backendIP}/users`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    id: userId,
                    is_allowed: currentUser.is_allowed, // Preserve current access status
                    low_security: !currentStatus, // Toggle low security mode
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to update security mode");
            }

            // Update local state
            setUsers((prevUsers) =>
                prevUsers.map((user) =>
                    user.id === userId
                        ? { ...user, low_security: !currentStatus }
                        : user
                )
            );

            Alert.alert(
                "Success", 
                `${!currentStatus ? "Low security" : "Standard security"} mode enabled for user`
            );
        } catch (error) {
            Alert.alert("Error", "Failed to update security mode");
        }
    };

    const handleScheduleCreated = () => {
        setShowSchedulePicker(false);
        fetchUsers(); // Refresh the user list
    };

    const updateUserName = async (userId: number, newName: string) => {
        try {
            // Find the current user to get their is_allowed status
            const currentUser = users.find(user => user.id === userId);
            if (!currentUser) {
                throw new Error("User not found");
            }

            const response = await fetch(`${backendIP}/users`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    id: userId,
                    name: newName,
                    is_allowed: currentUser.is_allowed, // Include current is_allowed status
                    low_security: currentUser.low_security // Preserve low security setting
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to update user name");
            }

            // Update local state
            setUsers((prevUsers) =>
                prevUsers.map((user) =>
                    user.id === userId
                        ? { ...user, name: newName }
                        : user
                )
            );

            setEditingUser(null);
            Alert.alert("Success", "User name updated successfully");
        } catch (error) {
            Alert.alert("Error", "Failed to update user name");
        }
    };
    
    // Add keyboard listeners
    useEffect(() => {
        const showListener = Keyboard.addListener(
            Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow',
            () => {
                setKeyboardVisible(true);
                // Re-scroll when keyboard appears to ensure edit field is visible
                if (editingUser !== null) {
                    setTimeout(() => scrollToEditingItem(editingUser), 100);
                }
            }
        );
        
        const hideListener = Keyboard.addListener(
            Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide',
            () => {
                setKeyboardVisible(false);
            }
        );
        
        return () => {
            showListener.remove();
            hideListener.remove();
        };
    }, [editingUser]);

    // Function to scroll to editing item
    const scrollToEditingItem = (userId: number) => {
        if (scrollViewRef.current && itemRefs.current[userId.toString()]) {
            // Find the y-position of the editing item
            const node = findNodeHandle(itemRefs.current[userId.toString()]);
            if (node) {
                // Calculate position to scroll to (adjust the 150 value as needed)
                scrollViewRef.current.scrollTo({
                    y: getItemPosition(userId) - 150,
                    animated: true
                });
            }
        }
    };

    // Helper function to estimate item position
    const getItemPosition = (userId: number) => {
        // Estimate position based on index
        const index = users.findIndex(user => user.id === userId);
        // Assuming each item is approximately 180px tall on average
        // Adjust the 180 and other values based on your actual UI
        return index * 180 + 250; // 250 accounts for the Add User section and padding
    };

    // Updated edit name press handler
    const handleEditNamePress = (user: User) => {
        setEditingUser(user.id);
        setEditName(user.name || '');
        
        // Wait a bit for state to update then scroll
        setTimeout(() => {
            scrollToEditingItem(user.id);
        }, 50);
    };

    if (loading) {
        return (
            <View style={styles.center}>
                <ActivityIndicator size="large" color="#0000ff" />
            </View>
        );
    }

    return (
        <KeyboardAvoidingView 
            style={{ flex: 1 }}
            behavior={Platform.OS === "ios" ? "padding" : "height"}
            keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 20}
        >
            <ScrollView 
                ref={scrollViewRef}
                style={styles.scrollContainer}
                contentContainerStyle={styles.scrollContent}
                keyboardShouldPersistTaps="handled"
            >
                <View style={styles.addUserSection}>
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
                    <Button title="Add User" onPress={addUser}/>
                </View>

                {/* User Management Section */}
                <Text style={[styles.title, {fontSize: 20, marginTop: 32}]}> User Management</Text>
                {users.map((item) => (
                    <View 
                        key={item.id.toString()} 
                        style={styles.userItem}
                    >
                        <TouchableOpacity 
                            style={styles.deleteButton}
                            onPress={() => {
                                Alert.alert(
                                    "Confirm Deletion",
                                    `Are you sure you want to remove ${item.name || "this user"}?`,
                                    [
                                        { text: "Cancel", style: "cancel" },
                                        { 
                                            text: "Delete", 
                                            style: "destructive",
                                            onPress: () => removeUser(item.id) 
                                        }
                                    ]
                                );
                            }}
                        >
                            <Text style={styles.deleteButtonText}>×</Text>
                        </TouchableOpacity>
                        <View style={styles.userInfo}>
                            {editingUser === item.id ? (
                                <View 
                                    style={styles.editNameContainer}
                                    ref={ref => {
                                        itemRefs.current[item.id.toString()] = ref;
                                    }}
                                >
                                    <TextInput 
                                        style={styles.editNameInput}
                                        value={editName}
                                        onChangeText={setEditName}
                                        autoFocus
                                    />
                                    <View style={styles.editNameButtons}>
                                        <TouchableOpacity
                                            style={[styles.editNameButton, styles.cancelButton]}
                                            onPress={() => setEditingUser(null)}
                                        >
                                            <Text style={styles.editButtonText}>Cancel</Text>
                                        </TouchableOpacity>
                                        <TouchableOpacity
                                            style={[styles.editNameButton, styles.saveButton]}
                                            onPress={() => updateUserName(item.id, editName)}
                                        >
                                            <Text style={styles.editButtonText}>Save</Text>
                                        </TouchableOpacity>
                                    </View>
                                </View>
                            ) : (
                                <TouchableOpacity 
                                    ref={ref => {
                                        itemRefs.current[item.id.toString()] = ref;
                                    }}
                                    onPress={() => handleEditNamePress(item)}
                                >
                                    <View style={styles.nameContainer}>
                                        <Text style={styles.userName}>
                                            {item.name || "Unnamed User"}
                                        </Text>
                                        <TouchableOpacity
                                            style={[
                                                styles.securityIcon,
                                                item.low_security ? styles.lowSecurityIcon : styles.standardSecurityIcon,
                                            ]}
                                            onPress={() => toggleLowSecurity(item.id, item.low_security)}
                                        >
                                            <Ionicons 
                                                name={item.low_security ? "lock-open" : "lock-closed"} 
                                                size={16} 
                                                color="white" 
                                            />
                                        </TouchableOpacity>
                                    </View>
                                </TouchableOpacity>
                            )}
                            <Text style={styles.phoneNumber}>{item.phone_number}</Text>
                        </View>
                        <View style={styles.actions}>
                            <View style={styles.actionButtons}>
                                <TouchableOpacity
                                    style={[
                                        styles.actionButton,
                                        item.is_allowed ? styles.allowed : styles.notAllowed,
                                    ]}
                                    onPress={() => toggleUserAccess(item.id, item.is_allowed)}
                                >
                                    <Text style={styles.buttonText}>
                                        {item.is_allowed ? "Revoke Access" : "Grant Access"}
                                    </Text>
                                </TouchableOpacity>
                                <TouchableOpacity
                                    style={styles.actionButton}
                                    onPress={() => {
                                        setSelectedUser(item);
                                        setShowSchedulePicker(true);
                                    }}
                                >
                                    <Text style={styles.buttonText}>Set Schedule</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>
                ))}
                {/* Add more padding at the bottom for keyboard */}
                <View style={{height: 200}} />
                
                {showSchedulePicker && selectedUser && (
                    <View style={styles.schedulePicker}>
                        <SchedulePicker
                            userId={selectedUser.id}
                            onScheduleCreated={handleScheduleCreated}
                        />
                    </View>
                )}
            </ScrollView>
        </KeyboardAvoidingView>
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
        flexDirection: "column",
        justifyContent: "space-between",
    },
    actionButtons: {
        flexDirection: "row",
        justifyContent: "space-between",
        gap: 8,
    },
    actionButton: {
        flex: 1,
        padding: 8,
        borderRadius: 4,
        alignItems: "center",
        backgroundColor: "#007AFF",
    },
    allowed: {
        backgroundColor: "#ff3b30",
    },
    notAllowed: {
        backgroundColor: "#34c759",
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
    deleteButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: '#f8f8f8',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 10,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.2,
        shadowRadius: 1,
    },
    deleteButtonText: {
        fontSize: 20,
        fontWeight: "bold",
        color: "#ff3b30",
        textAlign: 'center',
        marginTop: -2,
    },
    editNameContainer: {
        marginBottom: 8,
    },
    editNameInput: {
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 4,
        padding: 6,
        fontSize: 16,
    },
    editNameButtons: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        marginTop: 4,
    },
    editNameButton: {
        paddingVertical: 4,
        paddingHorizontal: 8,
        borderRadius: 4,
        marginLeft: 8,
    },
    cancelButton: {
        backgroundColor: '#ccc',
    },
    saveButton: {
        backgroundColor: '#34c759',
    },
    editButtonText: {
        color: '#fff',
        fontWeight: 'bold',
    },
    scrollContainer: {
        flex: 1,
    },
    scrollContent: {
        padding: 16,
    },
    addUserSection: {
        marginBottom: 16,
    },
    nameContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    securityIcon: {
        width: 24,
        height: 24,
        borderRadius: 12,
        justifyContent: 'center',
        alignItems: 'center',
    },
    lowSecurityIcon: {
        backgroundColor: "#ff3b30",
    },
    standardSecurityIcon: {
        backgroundColor: "#34c759",
    },
    schedulePicker: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        zIndex: 100,
        elevation: 5,
        padding: 16,
    },
});