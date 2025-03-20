import React, { useState, useEffect } from "react";
import { View, ScrollView, Switch, Alert, ActivityIndicator } from "react-native";
import { Button, Card, Text } from "react-native-paper";
import DateTimePicker from "@react-native-community/datetimepicker";

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;
const API_URL = `${backendIP}/schedule`;

const daysOfWeek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

type ScheduleItem = {
  open: Date | null;
  close: Date | null;
  forceUnlocked: boolean;
};

const defaultSchedule: ScheduleItem[] = daysOfWeek.map(() => ({
  open: null,
  close: null,
  forceUnlocked: false,
}));

const SchedulePage = () => {
  const [schedule, setSchedule] = useState<ScheduleItem[]>(defaultSchedule);
  const [pickerVisible, setPickerVisible] = useState<{ index: number | null; type: "open" | "close" | null }>({
    index: null,
    type: null,
  });
  const [loading, setLoading] = useState(true);

  // Fetch schedule from backend
  useEffect(() => {
    fetch(API_URL)
      .then(response => response.json())
      .then(data => {
        console.log("Fetched schedule data:", data);
        if (!Array.isArray(data) || data.length !== 7) {
          throw new Error("Invalid schedule format from backend");
        }
        const formattedData = data.map((item: any, index: number) => ({
          open: item.open ? new Date(item.open) : null,
          close: item.close ? new Date(item.close) : null,
          forceUnlocked: item.forceUnlocked ?? false,
        }));
        setSchedule(formattedData);
      })
      .catch(error => {
        console.error("Error fetching schedule:", error);
        Alert.alert("Error", "Failed to load schedule. Using default settings.");
        setSchedule(defaultSchedule);
      })
      .finally(() => setLoading(false));
  }, []);

  // Update open/close time
  const updateTime = (index: number, type: "open" | "close", time: Date) => {
    setSchedule(prevSchedule => {
      const newSchedule = [...prevSchedule];
      newSchedule[index] = { ...newSchedule[index], [type]: time };
      return newSchedule;
    });
  };

  // Toggle force unlock
  const toggleForceUnlock = (index: number) => {
    setSchedule(prevSchedule => {
      const newSchedule = [...prevSchedule];
      newSchedule[index] = { ...newSchedule[index], forceUnlocked: !newSchedule[index].forceUnlocked };
      return newSchedule;
    });
  };

  // Save schedule to backend
  const saveSchedule = () => {
    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        schedule.map(item => ({
          open: item.open ? item.open.toISOString() : null,
          close: item.close ? item.close.toISOString() : null,
          forceUnlocked: item.forceUnlocked,
        }))
      ),
    })
      .then(response => {
        if (!response.ok) throw new Error("Failed to save schedule");
        return response.json();
      })
      .then(() => Alert.alert("Success", "Schedule updated successfully"))
      .catch(error => {
        console.error("Error saving schedule:", error);
        Alert.alert("Error", "Failed to update schedule");
      });
  };

  // Show loading indicator while fetching data
  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text>Loading schedule...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={{ padding: 20 }}>
      {daysOfWeek.map((day, index) => (
        <Card key={day} style={{ marginBottom: 10, padding: 10 }}>
          <Text style={{ fontSize: 18, fontWeight: "bold" }}>{day}</Text>
          <View style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 10 }}>
            <Button mode="outlined" onPress={() => setPickerVisible({ index, type: "open" })}>
              {schedule[index]?.open ? schedule[index].open.toLocaleTimeString() : "Set Open Time"}
            </Button>
            <Button mode="outlined" onPress={() => setPickerVisible({ index, type: "close" })}>
              {schedule[index]?.close ? schedule[index].close.toLocaleTimeString() : "Set Close Time"}
            </Button>
          </View>
          <View style={{ flexDirection: "row", alignItems: "center", marginTop: 10 }}>
            <Text>Force Unlock:</Text>
            <Switch
              value={schedule[index]?.forceUnlocked ?? false}
              onValueChange={() => toggleForceUnlock(index)}
            />
          </View>
          {pickerVisible.index === index && pickerVisible.type && (
            <DateTimePicker
              value={schedule[index]?.[pickerVisible.type] || new Date()}
              mode="time"
              is24Hour={true}
              display="default"
              onChange={(_, selectedTime) => {
                if (selectedTime) updateTime(index, pickerVisible.type as "open" | "close", selectedTime);
                setPickerVisible({ index: null, type: null });
              }}
            />
          )}
        </Card>
      ))}
      <Button mode="contained" style={{ marginTop: 20, marginBottom: 40, padding: 10 }} onPress={saveSchedule}>
        Save Schedule
      </Button>
    </ScrollView>
  );
};

export default SchedulePage;
