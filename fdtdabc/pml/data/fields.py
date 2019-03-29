import copy


# A set of split fields for electric or magnetic field
class Fields:
    def __init__(self, field_x, field_y, field_z):
        self.xy = self._create_component(field_x)
        self.xz = self._create_component(field_x)
        self.yx = self._create_component(field_y)
        self.yz = self._create_component(field_y)
        self.zx = self._create_component(field_z)
        self.zy = self._create_component(field_z)

    def _create_component(self, base):
        component = copy.deepcopy(base)
        component.values.fill(0.0)
        return component
