import fire
import train
import test
import predict
import utils  

if __name__ == '__main__':
  fire.Fire({
    'train' : train.train,
    'test' : test.test,
    'predict' : predict.predict
  })

