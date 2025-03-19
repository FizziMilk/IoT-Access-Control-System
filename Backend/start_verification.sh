#!/bin/bash
# start_verification.sh

# Initiate OTP Verification
echo "Starting verification..."
curl -X POST "http://127.0.0.1:5000/start-verification" \
	-H "Content-Type: application/json" \
	-d '{"phone_number": "+447399963248"}'

echo -e "\nOTP sent! Please check your phone."

#Prompt the user to enter the OTP code
echo -n "Enter the OTP code: "
read otp_code

echo "Checking verification..."
curl -X POST "http://127.0.0.1:5000/check-verification" \
	-H "Content-Type: application/json" \
	-d "{\"phone_number\": \"+447399963248\", \"code\": \"$otp_code\"}"

echo "Printing logs"
curl -X GET "http://127.0.0.1:5000/access-logs"
