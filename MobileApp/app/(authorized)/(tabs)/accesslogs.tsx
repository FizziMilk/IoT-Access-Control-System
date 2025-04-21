import React, { useState, useCallback } from "react";
import { View, FlatList, Text, StyleSheet, ActivityIndicator, Alert } from "react-native";
import { useFocusEffect } from "@react-navigation/native";

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;
const logsURL = `${backendIP}/access-logs`;

type AccessLog = {
  user: string;
  user_name: string | null;
  method: string;
  status: string;
  timestamp: string;
};

export default function AccessLogs() {
  const [logs, setLogs] = useState<AccessLog[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchLogs = useCallback(() => {
    setLoading(true);
    fetch(logsURL)
      .then((res) => {
        if (!res.ok) {
          return res.json().then(err => {
            throw new Error(err.details || `HTTP error! status: ${res.status}`);
          });
        }
        return res.json();
      })
      .then((data: AccessLog[]) => setLogs(data))
      .catch((err) => {
        console.error("Error fetching access logs:", err);
        Alert.alert("Error", `Failed to load access logs: ${err.message}`);
      })
      .finally(() => setLoading(false));
  }, []);

  // Fetch logs every time the tab is focused
  useFocusEffect(fetchLogs);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#0000ff" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={logs}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={styles.logItem}>
            <Text style={styles.user}>{item.user_name || item.user}</Text>
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