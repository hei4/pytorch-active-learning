class BaseMaker:
    def get_train_set(self):
        return self.train_set

    def get_valid_set(self):
        return self.valid_set
    
    def get_test_set(self):
        return self.test_set
    
    def get_unlabel_set(self):
        return self.unlabel_set
    
    def get_grid_set(self):
        return self.grid_set