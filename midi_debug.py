import mido

port = mido.open_output('GestureSound 1')

# send test messages
port.send(mido.Message('control_change', control=74, value=100))  # CC74 filter
port.send(mido.Message('pitchwheel', pitch=4000))  # Pitch bend
port.send(mido.Message('control_change', control=91, value=100))  # CC91 reverb

print("Sent CC74, CC91, and Pitch Bend.")
port.close()
