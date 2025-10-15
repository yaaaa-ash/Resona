import mido
import time

# Name of your loopMIDI port
MIDI_PORT_NAME = 'GestureSound 1'

# Open the port
outport = mido.open_output(MIDI_PORT_NAME)

# Send a note on (middle C)
note = 60
velocity = 100
outport.send(mido.Message('note_on', note=note, velocity=velocity))
print("Note ON sent")

time.sleep(1)  # hold the note for 1 second

# Send note off
outport.send(mido.Message('note_off', note=note, velocity=0))
print("Note OFF sent")

outport.close()