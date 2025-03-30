import React, { useState, useEffect } from "react";
import { View, FlatList, Text, StyleSheet, ActivityIndicator, Alert} from "react-native";

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;
const logsURL = `${backendIP}/access-logs`

type AccessLog = {
    user: string;
    method: string;
    status: string;
    timestamp: string;
};

export default function AccessLogs() {
    const [logs, setLogs] = useState<AccessLog[]>([]);
    const [loading, setLoading] = useState(true);


    useEffect(() => {
        fetch(logsURL)
            .then((res) => res.json())
            .then((data: AccessLog[]) => setLogs(data))
            .catch((err) => {
                console.error("Error fetching access logs:", err);
                Alert.alert("Error", "Failed to load access logs.");
            })    
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <View style ={styles.center}>
                <ActivityIndicator size = "large" color="#0000ff" />
            </View>
        );
    }

    return (
        <View style = {styles.center}>
            <FlatList
            data={logs}
            keyExtractor = {(Item,index) => index.toString()}
            renderItem={({ item }) => (
                <View style = {styles.logItem}>
                    <Text style = {styles.user}> {item.user}</Text>
                    <Text style={styles.details}>
                        {item.method} - {item.status}
                    </Text>
                    <Text style={styles.timestamp}>{new Date(item.timestamp).toLocaleString()}</Text>
                </View>
            )}
            />
        </View>
    );
}
const styles = StyleSheet.create({
    container: {
      flex: 1,
      padding: 16,
      backgroundColor: "#fff",
    },
    center: {
      flex: 1,
      justifyContent: "center",
      alignItems: "center",
    },
    logItem: {
      marginBottom: 16,
      padding: 16,
      borderWidth: 1,
      borderColor: "#ddd",
      borderRadius: 8,
      backgroundColor: "#f9f9f9",
    },
    user: {
      fontWeight: "bold",
      fontSize: 16,
    },
    details: {
      fontSize: 14,
      color: "#555",
    },
    timestamp: {
      fontSize: 12,
      color: "#888",
    },
});