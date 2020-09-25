class Config(object):
    # Hyperparameters
    
    n_epochs = 100
    batch_size = 1000
    n_steps = 30
    
    n_clusters = 1 # number of clusters for GAN-clustering; currently no used
    dim_clusters = 64
    sigma_clusters = 1
    
    
    reg_strength_d = 0.0
    reg_strength_g = 0.0
    dropout_d = 0.0
    dropout_g = 0.0
    
    learning_rate_g = 0.001
    learning_rate_d = 0.001
    n_hidden_units_d = 300
    n_hidden_units_g = 300
    
    #leaky ReLU parameter
    alpha = 0.01
    
    def display():
        for i in dir(Config):
            if not i.startswith("_") and not callable(getattr(Config, i)):
                print(i + ": " + str(getattr(Config, i)))
            
            
