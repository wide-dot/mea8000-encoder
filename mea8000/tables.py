"""Parameter quantization tables of the MEA8000 (Philips TP101 table 2, clock 3.84 MHz)."""

FM1_HZ = (150, 162, 174, 188, 202, 217, 233, 250, 267, 286, 305, 325, 346, 368, 391, 415,
          440, 466, 494, 523, 554, 587, 622, 659, 698, 740, 784, 830, 880, 932, 988, 1047)

FM2_HZ = (440, 466, 494, 523, 554, 587, 622, 659, 698, 740, 784, 830, 880, 932, 988, 1047,
          1100, 1179, 1254, 1337, 1428, 1528, 1639, 1761, 1897, 2047, 2214, 2400, 2609, 2842, 3105, 3400)

FM3_HZ = (1179, 1337, 1528, 1761, 2047, 2400, 2842, 3400)

FM4_HZ = 3500

BW_HZ = (726, 309, 125, 50)

# amplitude * 1000, 3 dB per step
AMPL_PERMILLE = (0, 8, 11, 16, 22, 31, 44, 62, 88, 125, 177, 250, 354, 500, 707, 1000)

# pitch increment in Hz per 8 ms; code 16 selects the noise source
PI_HZ = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         0, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1)
NOISE_CODE = 16

FD_MS = (8, 16, 32, 64)

# TP101 table 2 footnote: exact pitch and pitch-increment values are the nominal ones x 1.024
PITCH_EXACT_SCALE = 1.024

# command register bits (TP101 table 5)
CMD_STOP = 0x10
CMD_CONT_ENABLE = 0x08
CMD_CONT = 0x04
CMD_ROE_ENABLE = 0x02
CMD_ROE = 0x01
CMD_POWER_ON = 0x1A  # slow-stop procedure, REQ pin disabled
