#[macro_use] extern crate cpython;

use cpython::{PyResult, Python, PyObject, PySequence, PyTuple};
use cpython::ObjectProtocol;

py_module_initializer!(rust2py, initrust2py, PyInit_rust2py, |py, m| {
    try!(m.add(py, "__doc__", "This module is implemented in Rust."));
    try!(m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64))));
    try!(m.add(py, "voxelize_cloud", py_fn!(py, py_voxelize_cloud(cloud: PyObject, k: f64))));
    Ok(())
});

// logic implemented as a normal rust function
fn sum_as_string(a:i64, b:i64) -> String {
    format!("{}", a + b).to_string()
}

// rust-cpython aware function. All of our python interface could be
// declared in a separate module.
// Note that the py_fn!() macro automatically converts the arguments from
// Python objects to Rust values; and the Rust return value back into a Python object.
fn sum_as_string_py(_: Python, a:i64, b:i64) -> PyResult<String> {
    let out = sum_as_string(a, b);
    Ok(out)
}

fn py_voxelize_cloud(py: Python, cloud: PyObject, k: f64) ->PyResult<i32> {
    let coords = cloud.getattr(py, "coords").unwrap().cast_into::<PySequence>(py).unwrap();
    let py_bb_min = cloud.getattr(py, "bb_min").unwrap().cast_into::<PySequence>(py).unwrap();
    let bb_min = (0.1, 0.1, 0.1);


    let num_points = coords.len(py).unwrap();
    for i in 0..num_points {
        let p = coords.get_item(py, i).unwrap();

        let x: f64 = p.get_item(py, 0).unwrap().extract(py).unwrap();
        let y: f64 = p.get_item(py, 1).unwrap().extract(py).unwrap();
        let z: f64 = p.get_item(py, 2).unwrap().extract(py).unwrap();
        ok(&(x, y, z), &(bb_min), k);

    }

    Ok(2)
}


fn ok(point: &(f64, f64, f64), bb_min: &(f64, f64, f64), k: f64) -> (i32, i32, i32) {
    (((point.0 - bb_min.0) / k) as i32,
     ((point.1 - bb_min.1) / k) as i32,
     ((point.2 - bb_min.2) / k) as i32)
}
