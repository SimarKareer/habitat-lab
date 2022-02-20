import numpy as np

from habitat.tasks.locomotion.aliengo import (
    AlienGo,
    VectorCachableProperty,
    attribute_to_str,
)


def vectorize_and_cache(attr):
    def wrapper(instance):
        if attr.cache_key not in instance.cache:
            matrix = []
            for robot in instance.robots:
                matrix.append(getattr(robot, attr.cache_key))
            instance.cache[attr.cache_key] = np.array(matrix)
        return instance.cache[attr.cache_key]

    return wrapper


def iterate_all(attr):
    def wrapper(instance, *args, **kwargs):
        attr_str = attribute_to_str(attr)
        for robot in instance.robots:
            getattr(robot, attr_str)(*args, **kwargs)

    return wrapper


def decorate_all_attributes(decorator1, decorator2):
    def decorate(cls):
        # Go through the methods of the super class
        for method in vars(cls.__base__):
            if method.startswith("__"):
                continue
            # Identify VectorCachableProperty attributes
            if isinstance(getattr(cls, method), VectorCachableProperty):
                # Apply input decorator to these attributes
                setattr(
                    cls, method, property(decorator1(getattr(cls, method)))
                )
            else:
                setattr(cls, method, decorator2(getattr(cls, method)))
        return cls

    return decorate


@decorate_all_attributes(vectorize_and_cache, iterate_all)
class AlienGoVectorized(AlienGo):
    cache = {}

    def __init__(
        self,
        robot_ids,
        sim,
        fixed_base,
        robot_cfg,
        reset_positions,
    ):
        # Call __init__ of super class to initialize parameters
        super().__init__(None, sim, fixed_base, robot_cfg)
        self.robots = [
            AlienGo(robot_id, sim, fixed_base, robot_cfg, reset_position)
            for robot_id, reset_position in zip(robot_ids, reset_positions)
        ]
        self.robot_ids = robot_ids
        self.reset_positions = reset_positions
