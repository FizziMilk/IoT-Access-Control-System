import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import { backendIP } from '../config';

type SchedulePickerProps = {
  userId: number;
  onScheduleCreated?: () => void;
};

export default function SchedulePicker({ userId, onScheduleCreated }: SchedulePickerProps) {
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [showStartPicker, setShowStartPicker] = useState(false);
  const [showEndPicker, setShowEndPicker] = useState(false);

  const handleCreateSchedule = async () => {
    try {
      const response = await fetch(`${backendIP}/user-schedule`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          start_date: startDate.toISOString(),
          end_date: endDate.toISOString(),
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to create schedule');
      }

      Alert.alert('Success', 'Schedule created successfully');
      onScheduleCreated?.();
    } catch (error) {
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to create schedule');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Set Access Schedule</Text>
      
      <View style={styles.dateContainer}>
        <Text style={styles.label}>Start Date:</Text>
        <TouchableOpacity 
          style={styles.dateButton}
          onPress={() => setShowStartPicker(true)}
        >
          <Text>{startDate.toLocaleDateString()}</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.dateContainer}>
        <Text style={styles.label}>End Date:</Text>
        <TouchableOpacity 
          style={styles.dateButton}
          onPress={() => setShowEndPicker(true)}
        >
          <Text>{endDate.toLocaleDateString()}</Text>
        </TouchableOpacity>
      </View>

      {showStartPicker && (
        <DateTimePicker
          value={startDate}
          mode="date"
          display="default"
          onChange={(event, selectedDate) => {
            setShowStartPicker(false);
            if (selectedDate) {
              setStartDate(selectedDate);
            }
          }}
          minimumDate={new Date()}
        />
      )}

      {showEndPicker && (
        <DateTimePicker
          value={endDate}
          mode="date"
          display="default"
          onChange={(event, selectedDate) => {
            setShowEndPicker(false);
            if (selectedDate) {
              setEndDate(selectedDate);
            }
          }}
          minimumDate={startDate}
        />
      )}

      <TouchableOpacity 
        style={styles.createButton}
        onPress={handleCreateSchedule}
      >
        <Text style={styles.buttonText}>Create Schedule</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 8,
    marginVertical: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  dateContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  label: {
    width: 80,
    fontSize: 16,
  },
  dateButton: {
    flex: 1,
    padding: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
  },
  createButton: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 4,
    alignItems: 'center',
    marginTop: 16,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
}); 