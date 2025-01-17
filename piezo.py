import usb
import time

class Piezo():
    def __init__(self):
        dev = usb.core.find(idVendor=0x104d, idProduct=0x4000)
        cfg = dev.get_active_configuration()
        intf = cfg[(0,0)]

        self.ep_out = usb.util.find_descriptor(intf,
            custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) ==
                usb.util.ENDPOINT_OUT)
        assert self.ep_out is not None
        assert self.ep_out.wMaxPacketSize == 64
        self.ep_in = usb.util.find_descriptor(intf,
            custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) ==
                usb.util.ENDPOINT_IN)
        assert self.ep_in is not None
        assert self.ep_in.wMaxPacketSize == 64


        self.write('*IDN?')
        print(self.read())

    def write(self, command):
        self.ep_out.write(command.encode() + b'\r\n')

    def read(self):
        r = self.ep_in.read(64).tobytes()
        return r

    def move_to(self, axis, location): self.write(f'{axis}PA{location}')
    def move_by(self, axis, delta): self.write(f'{axis}PR{delta}')

    def set_v(self, axis, v): self.write(f'{axis}VA{v};SM')

    def get_position(self, axis):
        self.write(f'{axis}PA?')
        return int(self.read()[:-2])
