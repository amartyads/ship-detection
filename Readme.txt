This project works on the dataset available at https://www.kaggle.com/rhammell/ships-in-satellite-imagery .

The code uses CNNs to classify images. In one approach, the images are written into tfrecord files, which are then used for training and testing. In the second approach, the json file provided acts as the source.

The code was written using Spyder and requires tensorflow to run.

Please download the dataset available at the link and paste the data in a subfolder. Edit the source and dest variables accordingly.

tfrecords are available, so the images need not be present, as the shipclassifier.py works solely on the records. shipclassifierjson.py requires the json file, however.