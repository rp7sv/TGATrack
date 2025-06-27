class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/2'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/2/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/2/pretrained_networks'
        self.got10k_val_dir = '/data/2/data/got10k/val'
        self.lasot_lmdb_dir = '/data/2/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/2/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/2/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/2/data/coco_lmdb'
        self.coco_dir = '/data/2/data/coco'
        self.lasot_dir = '/data/2/data/lasot'
        self.got10k_dir = '/data/2/data/got10k/train'
        self.trackingnet_dir = '/data/2/data/trackingnet'
        self.depthtrack_dir = '/data/2/data/depthtrack/train'
        self.lasher_dir = '/data/lasher/trainingset'
        self.visevent_dir = '/data/2/data/visevent/train'
