class DoomEnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """
        #import pdb;pdb.set_trace()
        from doom_emulator import DoomEmulator
        #from ale_python_interface import ALEInterface
        #filename = args.rom_path + "/" + args.game + ".bin"
        #ale_int = ALEInterface()
        #ale_int.loadROM(str.encode(filename))
        #self.num_actions = len(ale_int.getMinimalActionSet())
        
        # TODO: do better
        self.num_actions = 3
        self.create_environment = lambda i: DoomEmulator(i, args)
