from Preprocessor import Preprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.STL10 import STL10
from models.AutoEncoder import AutoEncoder
from models.SpotNet import SNet
from layers import lrelu

target_shape = [96, 96, 3]
ae = AutoEncoder(num_layers=4, batch_size=128, target_shape=target_shape, activation_fn=lrelu, tag='default')
model = SNet(ae, batch_size=128, target_shape=target_shape, disc_pad='SAME', tag='default')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, init_lr=0.0003,
                       lr_policy='linear', num_gpus=2)
trainer.train_model(None)
