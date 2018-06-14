from Preprocessor import Preprocessor
from train.ClassifierTrainer import ClassifierTrainer
from datasets.STL10 import STL10
from models.SpotNet import SNet
from layers import lrelu

target_shape = [96, 96, 3]
model = SNet(None, batch_size=128, target_shape=target_shape, disc_pad='SAME', tag='default')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = ClassifierTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, optimizer='adam',
                            lr_policy='linear', init_lr=0.001, num_gpus=1, num_conv2train=5, num_conv2init=5)
trainer.train_model(None)
