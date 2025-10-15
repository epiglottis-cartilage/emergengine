use std::ffi::CStr;

pub const ENGINE_VERSION: [u32; 4] = const {
    const VERSION: &str = env!("CARGO_PKG_VERSION");
    let bytes = VERSION.as_bytes();

    let mut result = [0; 4];
    let mut current = 0;
    let mut part_index = 0;

    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            c if c.is_ascii_digit() => {
                current = current * 10 + (c - b'0') as u32;
            }
            _ => {
                if part_index < 4 {
                    result[part_index] = current;
                    part_index += 1;
                    current = 0;
                }
            }
        }
        i += 1;
    }
    if part_index < 4 {
        result[part_index] = current;
    }
    result
};

/// Get required instance extensions.
/// This is windows specific.
#[allow(unused_imports)]
pub fn required_extension_names() -> Vec<&'static CStr> {
    use ash::ext::debug_utils::NAME as DebugUtilsName;
    use ash::khr::surface::NAME as SurfaceName;
    use ash::khr::win32_surface::NAME as Win32SurfaceName;
    let mut res = vec![SurfaceName];
    if cfg!(debug_assertions) {
        res.push(DebugUtilsName);
    }
    res
}
