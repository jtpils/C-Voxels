import math
import laspy
import numpy as np


class GenericPoint:
    def __init__(self, *args, **dimensions):
        if len(args) == 1:
            args = args[0]
        self.x, self.y, self.z = args[0], args[1], args[2]
        for dimension_name, dimension_value in dimensions.items():
            setattr(self, dimension_name, dimension_value)

    @property
    def coords(self):
        return self.x, self.y, self.z

    def __iter__(self):
        iter(self.__dict__.values())

    def __repr__(self):
        return "GenericPoint({})".format(", ".join("{}={}".format(dim, value) for dim, value in self.__dict__.items()))


class PointCloud(object):
    def __init__(self, coords, classification, bb_min=None, bb_max=None, intensity=None, rgb=None, gps_time=None,
                 **dimensions):
        self.coords = np.double(coords, dtype=np.double)
        self.classification = classification
        self.intensity = intensity
        self.rgb = rgb
        self.gps_time = gps_time
        self.bb_min = np.min(self.coords, axis=0) if bb_min is None else bb_min
        self.bb_max = np.max(self.coords, axis=0) if bb_max is None else bb_max
        for dimension_name, dimension_value in dimensions.items():
            setattr(self, dimension_name, dimension_value)

    def update_bbox(self):
        self.bb_min = np.min(self.coords, axis=0)
        self.bb_max = np.max(self.coords, axis=0)

    def write(self, file_name):
        with open(file_name, 'w') as out_file:
            for (i, point) in enumerate(self.coords):
                out_file.write("{} {} {} {}\n".format(point[0], point[1], point[2], self.classification[i]))

    def reset_classification(self, codes=None):
        if codes:
            for code in codes:
                self.classification[self.classification == code] = 0
        self.classification[:] = 0

    def has_class(self, code):
        return np.any(self.classification == code)


    @staticmethod
    def invalid_dimensions():
        return {"coords", "bb_min", "bb_max"}

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, item):
        dimensions = (dim for dim in self.__dict__.keys() if dim not in self.__class__.invalid_dimensions())
        if isinstance(item, int):
            point = GenericPoint(
                self.coords[item],
                **{dim: self.__dict__[dim][item] for dim in dimensions if self.__dict__[dim] is not None})
            return point

        if isinstance(item, slice) or len(item) > 1:
            new_pc = PointCloud(
                self.coords[item],
                **{dim: self.__dict__[dim][item] for dim in dimensions if self.__dict__[dim] is not None}
            )
            return new_pc


class LasCloud(PointCloud):
    def __init__(self, filename, mode="rw"):
        self.las_file = laspy.file.File(filename, mode=mode)
        coords = np.vstack((self.las_file.x, self.las_file.y, self.las_file.z)).transpose()
        super(LasCloud, self).__init__(
            coords,
            self.las_file.get_classification(),
            self.las_file.header.get_min(),
            self.las_file.header.get_max(),
            self.las_file.get_intensity(),
            self.set_rgb(),
            self.set_gps()
        )

    def set_rgb(self):
        rgb_values = np.zeros((len(self.las_file), 3), dtype=np.uint8)
        color_channels = ['red', 'green', 'blue']
        try:
            if sum([np.linalg.norm(getattr(self.las_file, color)) == 0 for color in color_channels]) == len(
                    color_channels):
                return None
            rgb_values[:, 0] = np.array([red >> 8 for red in self.las_file.red], dtype=np.uint8)
            rgb_values[:, 1] = np.array([green >> 8 for green in self.las_file.green], dtype=np.uint8)
            rgb_values[:, 2] = np.array([blue >> 8 for blue in self.las_file.blue], dtype=np.uint8)
            return rgb_values
        except laspy.util.LaspyException:
            return None

    def set_gps(self):
        try:
            return self.las_file.gps_time
        except laspy.util.LaspyException:
            return None

    def write(self, filename):

        hdr = laspy.header.Header()

        outfile = laspy.file.File(filename, mode="w", header=hdr)
        all_coords = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        mins = map(np.min, all_coords)

        outfile.header.offset = list(mins)
        outfile.header.scale = [0.001, 0.001, 0.001]

        outfile.header.min = [np.min(self.coords[:, i]) for i in range(3)]
        outfile.header.max = [np.max(self.coords[:, i]) for i in range(3)]

        outfile.x, outfile.y, outfile.z = all_coords
        outfile.classification = self.classification
        setattr(outfile, "intensity", self.intensity)
        outfile.close()

    def write_object(self, filename, indices):
        points_kept = self.las_file.points[indices]
        mins = [np.min(axis) for axis in [self.las_file.X, self.las_file.Y, self.las_file.Z]]
        output_file = laspy.file.File(filename, mode="w", header=self.las_file.header)
        output_file.header.offset = mins
        output_file.points = points_kept
        output_file.close()

    def save_classification(self):
        self.las_file.classification = self.classification

    def save_colors(self):
        self.las_file.red = self.rgb[:, 0]
        self.las_file.green = self.rgb[:, 1]
        self.las_file.blue = self.rgb[:, 2]

    def close(self):
        self.las_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rgb is not None:
            self.save_colors()
        self.save_classification()
        self.las_file.close()

    def __repr__(self):
        return "{}({}): {} points".format(self.__class__.__name__, self.las_file.filename, len(self))

    @staticmethod
    def invalid_dimensions():
        d = {'las_file'}
        d.update(PointCloud.invalid_dimensions())
        return d