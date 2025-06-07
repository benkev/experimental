'''
receive_ble_adc.py - receive streaming ADC data over BLE from ESP32 to a PC

'''

import asyncio
from bleak import BleakClient, BleakScanner

UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

def handle_notify(_, data):
    value = int.from_bytes(data, byteorder='little')
    print("ADC:", value)

async def main():
    devices = await BleakScanner.discover()
    for d in devices:
        if "ESP32-BLE-ADC" in d.name:
            async with BleakClient(d.address) as client:
                await client.start_notify(UUID, handle_notify)
                await asyncio.sleep(10)
                await client.stop_notify(UUID)

asyncio.run(main())
