#
# Sending AT commands via the serial port to the Bluetooth transceiver
# module xs3868 based on the OVC3860 chip.
#
import serial

ser = serial.Serial(port='COM5', baudrate=115200, timeout=.01)

print 'ser.isOpen() = ', ser.isOpen()

# Enter pairing
ser.write("AT#CA\r\n"); print '"AT#CA" -> ', ser.read(128)
# Cancel pairing
ser.write("AT#CB\r\n"); print '"AT#CB" -> ', ser.read(128)
# Query status
ser.write("AT#CY\r\n"); print '"AT#CY" -> ', ser.read(128)
# Reset
ser.write("AT#CZ\r\n"); print '"AT#CZ" -> ', ser.read(128)
# Query status
ser.write("AT#CY\r\n"); print '"AT#CY" -> ', ser.read(128)
# Play/Pause
ser.write("AT#MA\r\n"); print '"AT#MA" -> ', ser.read(128)
# Stop
ser.write("AT#MC\r\n"); print '"AT#MC" -> ', ser.read(128)
# Connect to av source
ser.write("AT#MI\r\n"); print '"AT#MI" -> ', ser.read(128)
# Disconnect from av source
ser.write("AT#MJ\r\n"); print '"AT#MJ" -> ', ser.read(128)
# Query avrcp status
ser.write("AT#MO\r\n"); print '"AT#MO" -> ', ser.read(128)
# Start FF
ser.write("AT#MR\r\n"); print '"AT#MR" -> ', ser.read(128)
# Start Rewind
ser.write("AT#MS\r\n"); print '"AT#MS" -> ', ser.read(128)
# Stop FF/Rewind
ser.write("AT#MT\r\n"); print '"AT#MT" -> ', ser.read(128)
# Query A2DP Status
ser.write("AT#MV\r\n"); print '"AT#MV" -> ', ser.read(128)
# Decrease Volume
ser.write("AT#VD\r\n"); print '"AT#VD" -> ', ser.read(128)
# Increase Volume
ser.write("AT#VU\r\n"); print '"AT#VU" -> ', ser.read(128)
# Read a byte from memory
ser.write("AT#MX08001AC4\r\n"); print '"AT#MX08001AC4" -> ', ser.read(128)

# Print 64 bytes of memory starting from 0x08000000
for a in xrange(64):
    addr = 0x08000000 + a 
    ser.write("AT#MX"+hex(addr)+"\r\n")
    r = ser.read(128) 
    i = r.find('MEM:')  # Index of first letter of "MEM:" in r
    r = r[i+4:]           # Cut away up to end of 'MEM:'
    i = r.find('\r')    # Index of first letter "\r" in r
    # In hexadecimal, 2 digits with leading zero:
    print '%08x: %02x ' % (addr, int(r[:i], base=16))     


ser.close()

