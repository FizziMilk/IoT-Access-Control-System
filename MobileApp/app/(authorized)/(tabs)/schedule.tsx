import React, { useState, useEffect } from "react";
import { View, ScrollView, Switch, Alert, ActivityIndicator } from "react-native";
import { Text, Button, Card } from "react-native-paper";
import DateTimePicker from "@react-native-community/datetimepicker";

const backendIP = process.env.EXPO_PUBLIC_BACKEND_IP;
const backendURL = `${backendIP}/schedule`;

type ScheduleEntry = {
  day: string;
  open_time: string | null;     // e.g. "08:00"
  close_time: string | null;    // e.g. "18:00"
  forceUnlocked: boolean;
};

// For local state, we store them as Dates for easy DateTimePicker usage.
type LocalEntry = {
  day: string;
  open: Date | null;
  close: Date | null;
  forceUnlocked: boolean;
};

const timeStringToDate = (timeString: string): Date => {
  const [hours, minutes] = timeString.split(':').map(Number);
  const date = new Date();
  date.setHours(hours, minutes, 0, 0);
  return date;
};

const SchedulePage = () => {
  const [schedule, setSchedule] = useState<LocalEntry[]>([]);
  const [loading, setLoading] = useState(true);

  // Which day/time are we editing?
  const [pickerVisible, setPickerVisible] = useState<{
    index: number;
    type: "open" | "close";
  } | null>(null);

  // ─────────────────────────────────────────────────────────────────────────────
  // 1. Load the Current Schedule from the Backend
  // ─────────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(backendURL)
      .then((res) => res.json())
      .then((data: ScheduleEntry[]) => {
      // Convert schedule to local state with Date objects
        const converted = data.map((item) => ({
          day: item.day,
          open: item.open_time ? timeStringToDate(item.open_time) : null,
          close: item.close_time ? timeStringToDate(item.close_time) : null,
          forceUnlocked: item.forceUnlocked,
        }));
        setSchedule(converted);
      })
      .catch((err) => {
        console.error("Error fetching schedule:", err);
        Alert.alert("Error", "Failed to load schedule.");
      })
      .finally(() => setLoading(false));
  }, []);

  // ─────────────────────────────────────────────────────────────────────────────
  // 2. Save the Updated Schedule to the Backend
  // ─────────────────────────────────────────────────────────────────────────────
  const saveSchedule = () => {
    // Convert local state (Dates) back to strings ("HH:MM")
    const payload: ScheduleEntry[] = schedule.map((entry) => ({
      day: entry.day,
      open_time: entry.open
        ? entry.open.toTimeString().slice(0, 5) // e.g. "08:00"
        : null,
      close_time: entry.close
        ? entry.close.toTimeString().slice(0, 5)
        : null,
      forceUnlocked: entry.forceUnlocked,
    }));

    fetch(backendURL, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((res) => res.json())
      .then((result) => {
        console.log("Schedule updated:", result);
        Alert.alert("Success", "Schedule updated successfully!");
      })
      .catch((err) => {
        console.error("Error saving schedule:", err);
        Alert.alert("Error", "Failed to update schedule.");
      });
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // 3. Time Picker for "open" or "close"
  // ─────────────────────────────────────────────────────────────────────────────
  const handleTimeChange = (
    event: any,
    selectedTime: Date | undefined
  ) => {
    if (!pickerVisible) return;
    const { index, type } = pickerVisible;

    if (selectedTime) {
      const updated = [...schedule];
      updated[index][type] = selectedTime;
      setSchedule(updated);
    }
    // Hide the picker
    setPickerVisible(null);
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // UI
  // ─────────────────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator size="large" color="#000" />
        <Text>Loading schedule...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={{ padding: 20 }}>
      {schedule.map((entry, index) => (
        <Card key={entry.day} style={{ marginBottom: 10 }}>
          <Card.Title title={entry.day} />
          <Card.Content>
            <Text>Open Time:</Text>
            <Button
              mode="outlined"
              onPress={() => setPickerVisible({ index, type: "open" })}
            >
              {entry.open
                ? entry.open.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                : "Set Open Time"}
            </Button>

            <Text style={{ marginTop: 10 }}>Close Time:</Text>
            <Button
              mode="outlined"
              onPress={() => setPickerVisible({ index, type: "close" })}
            >
              {entry.close
                ? entry.close.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                : "Set Close Time"}
            </Button>

            <View style={{ flexDirection: "row", alignItems: "center", marginTop: 10 }}>
              <Text>Force Unlocked:</Text>
              <Switch
                value={entry.forceUnlocked}
                onValueChange={() => {
                  const updated = [...schedule];
                  updated[index].forceUnlocked = !updated[index].forceUnlocked;
                  setSchedule(updated);
                }}
                style={{ marginLeft: 10 }}
              />
            </View>
          </Card.Content>
        </Card>
      ))}

      {/* DateTimePicker for the currently editing day/time */}
      {pickerVisible && (
        <DateTimePicker
          mode="time"
          value={schedule[pickerVisible.index][pickerVisible.type] || new Date()}
          onChange={handleTimeChange}
          is24Hour
        />
      )}

      <Button
        mode="contained"
        style={{ marginTop: 20, padding: 10 }}
        onPress={saveSchedule}
      >
        Save Schedule
      </Button>
    </ScrollView>
  );
};

export default SchedulePage;
