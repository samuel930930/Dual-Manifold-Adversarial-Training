

class DotDict(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DotDict(value)
        return value



if __name__ == '__main__':
    import yaml

    config_file = './invGAN.yml'
    with open(config_file) as f:
        config = DotDict(yaml.load(f))

    print(config)
    print(config.generator)
    print(config.generator.type)
    print(config.generator.norm.bn)
    print(config.optimizer.type)
    #print(config.generator.a)