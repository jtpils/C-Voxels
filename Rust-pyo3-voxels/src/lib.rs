#![feature(proc_macro, specialization)]

extern crate pyo3;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::LinkedList;

#[py::modinit(rust2py)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "sum_as_string")]
    // pyo3 aware function. All of our python interface could be declared in a separate module.
    // Note that the `#[pyfn()]` annotation automatically converts the arguments from
    // Python objects to Rust values; and the Rust return value back into a Python object.
    fn sum_as_string_py(a:i64, b:i64) -> PyResult<String> {
       let out = sum_as_string(a, b);
       Ok(out)
    }

    #[pyfn(m, "voxelize_cloud")]
    fn py_voxelize_cloud(py: Python, cloud: PyObject, k: f64) -> PyResult<HashMap<(i32, i32, i32), i32>> {
        println!("Start getting attributes");
        let py_coords: PyObject = cloud.getattr(py, "coords")?;
        let py_bb_min: PyObject = cloud.getattr(py, "bb_min")?;

        let coords: &PySequence = py_coords.extract(py)?;
        let mins: Vec<f64> = py_bb_min.extract(py).unwrap();


        println!("ok");
        let voxels = voxelize_cloud(coords, &mins, k);
        println!("ok");

        Ok(voxels)
    }


    #[pyfn(m, "voxelize_cloud_2")]
    fn voxelize_cloud_2(coords: Vec<Vec<f64>>, bb_min: Vec<f64>, k: f64) -> PyResult<HashMap<(i32, i32, i32), i32>> {
        println!("ok");
        let voxels = voxelize(&coords, &bb_min, k);
        println!("ok");
        Ok(voxels)
    }

    Ok(())

}

fn voxelize_cloud(coords: &PySequence, bb_min: &Vec<f64>, k: f64) -> HashMap<(i32, i32, i32), i32> {
    let mut voxels = HashMap::<(i32, i32, i32), i32>::new();
    for i in 0..coords.len().unwrap() {
        let py_coords: &PyObjectRef = coords.get_item(i).unwrap();

        let vxl = ok(&point, &bb_min, k);

        let voxel = voxels.entry(vxl).or_insert(0);
        *voxel += 1;
    }
    voxels
}

fn ok(point: &(f64, f64, f64), bb_min: &Vec<f64>, k: f64) -> (i32, i32, i32) {
    (((point.0 - bb_min[0]) / k) as i32,
     ((point.1 - bb_min[1]) / k) as i32,
     ((point.2 - bb_min[2]) / k) as i32)
}

fn voxel_coordinates(point: &Vec<f64>, bb_min: &Vec<f64>, k: f64) -> (i32, i32, i32) {
    (((point[0] - bb_min[0]) / k) as i32,
     ((point[1] - bb_min[1]) / k) as i32,
     ((point[2] - bb_min[2]) / k) as i32)
}

fn voxelize(coords: &Vec<Vec<f64>>, bb_min: &Vec<f64>, k: f64) -> HashMap<(i32,i32,i32), i32> {
    let mut voxels = HashMap::<(i32, i32, i32), i32>::new();

    for point in coords {
        let voxel_coord = voxel_coordinates(point, bb_min, k);
        let voxel = voxels.entry(voxel_coord).or_insert(0);
        *voxel += 1;
    }
    voxels
}

// logic implemented as a normal rust function
fn sum_as_string(a:i64, b:i64) -> String {
    format!("{}", a + b).to_string()
}


