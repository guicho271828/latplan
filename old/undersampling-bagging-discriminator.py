# imbalanced data WIP ###############################################################

# In general, there are more invalid data than valid data. These kinds of
# imbalanced datasets always make it difficult to train a classifier.
# Theoretically, the most promising way for this problem is Undersampling + bagging.
# Yeah I know, I am not a statistician. But I have a true statistician friend !
# TBD : add reference to that paper (I forgot).

# Ultimately this implementation was not used during AAAI submission.

class UBDiscriminator(Discriminator):
    def _build(self,input_shape):
        x = Input(shape=input_shape)

        self.discriminators = []
        for i in range(self.parameters["bagging"]):
            d = Discriminator(self.path+"/"+str(i),self.parameters)
            d.build(input_shape)
            self.discriminators.append(d)

        y = average([ d.net(x) for d in self.discriminators ])
        y = wrap(y,K.round(y))
        self.net = Model(x,y)
        self.net.compile(optimizer="adam",loss=bce)

    def train(self,train_data,
              train_data_to=None,
              val_data=None,
              val_data_to=None,
              *args,**kwargs):

        self.build(train_data.shape[1:])

        num   = len(val_data_to)
        num_p = np.count_nonzero(val_data_to)
        num_n = num-num_p
        assert num_n > num_p
        print("positive : negative = ",num_p,":",num_n,"negative ratio",num_n/num_p)

        ind_p = np.where(val_data_to == 1)[0]
        ind_n = np.where(val_data_to == 0)[0]

        from numpy.random import shuffle
        shuffle(ind_n)

        per_bag = num_n // len(self.discriminators)
        for i, d in enumerate(self.discriminators):
            print("training",i+1,"/",len(self.discriminators),"th discriminator")
            ind_n_per_bag = ind_n[per_bag*i:per_bag*(i+1)]
            ind = np.concatenate((ind_p,ind_n_per_bag))
            d.train(train_data[ind],
                    train_data_to=train_data_to[ind],
                    val_data=val_data,
                    val_data_to=val_data_to,
                    *args,**kwargs)

    def discriminate(self,data,**kwargs):
        return self.net.predict(data,**kwargs)

