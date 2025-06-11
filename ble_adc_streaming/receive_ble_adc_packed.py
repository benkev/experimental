import asyncio
from bleak import BleakClient, BleakScanner

# UUIDs must match what your ESP32 is using
SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
CHAR_UUID    = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

def handle_notification(sender, data):
    # `data` is a bytes object
    value = int.from_bytes(data, byteorder='little')
    print(f"ADC value: {value}")

async def main():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()

    # Find ESP32 device advertising the desired service
    esp_device = None
    for d in devices:
        print(f"Found: {d.name} [{d.address}]")
        if d.name and "ESP32" in d.name:
            esp_device = d
            break

    if not esp_device:
        print("ESP32 device not found.")
        return

    print(f"Connecting to {esp_device.name} [{esp_device.address}]")

    async with BleakClient(esp_device.address) as client:
        print("Connected. Discovering services...")
        # services = await client.get_services()
        for s in client.services:
            print(f"- Service {s.uuid}")
            for c in s.characteristics:
                print(f"  - Char {c.uuid}")

        print("Subscribing to notifications...")
        await client.start_notify(CHAR_UUID, handle_notification)

        try:
            print("Receiving BLE notifications. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Disconnecting...")
            await client.stop_notify(CHAR_UUID)

if __name__ == "__main__":
    asyncio.run(main())
