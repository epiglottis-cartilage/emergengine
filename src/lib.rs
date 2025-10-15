mod error;
mod utils;
#[cfg(debug_assertions)]
pub use error::Result;

use ash::vk;
use std::ffi::{CStr, CString, c_void};

#[allow(unused)]
pub struct App {
    entry: ash::Entry,
    instance: ash::Instance,
    #[cfg(debug_assertions)]
    debug_utils: (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT),
}

#[cfg(debug_assertions)]
const REQUIRED_LAYERS: &[&CStr] =
    &[unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];

impl App {
    pub fn new() -> Result<Self> {
        log::info!("Creating application");

        let entry = unsafe { ash::Entry::load() }.expect("Fail to create entry");
        let instance = Self::create_instance(&entry)?;
        Ok(Self {
            #[cfg(debug_assertions)]
            debug_utils: Self::setup_debug_messenger(&entry, &instance)?,
            entry,
            instance, // TODO: fix typo
        })
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
        let app_name = CString::new("new window").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let [major, minor, patch, variant] = utils::ENGINE_VERSION;
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .api_version(ash::vk::make_api_version(variant, major, minor, patch))
            .engine_name(engine_name.as_c_str())
            .api_version(ash::vk::API_VERSION_1_3);

        let extension_names = utils::required_extension_names()
            .into_iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();

        let layer_name;
        if cfg!(debug_assertions) {
            layer_name = Self::check_validation_layer_support(entry)?
                .map(|layer| layer.as_ptr())
                .collect::<Vec<_>>();
        } else {
            layer_name = Vec::new();
        }

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_name);

        unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(Into::into)
        }
    }

    #[cfg(debug_assertions)]
    fn check_validation_layer_support(entry: &ash::Entry) -> Result<impl Iterator<Item = &CStr>> {
        let layers = unsafe { entry.enumerate_instance_layer_properties() }?;
        Ok(REQUIRED_LAYERS.iter().cloned().filter(move |&name| {
            if layers.iter().any(|layer_properties| {
                layer_properties
                    .layer_name_as_c_str()
                    .expect("Invalid Layer name")
                    == name
            }) {
                true
            } else {
                log::warn!("Validation layer not found :{}", name.to_str().unwrap());
                false
            }
        }))
    }

    #[cfg(debug_assertions)]
    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
            .pfn_user_callback(Some(Self::debug_callback));

        let reporter = ash::ext::debug_utils::Instance::new(entry, instance);
        let reporter_callback =
            unsafe { reporter.create_debug_utils_messenger(&create_info, None)? };

        Ok((reporter, reporter_callback))
    }
    unsafe extern "system" fn debug_callback(
        flag: vk::DebugUtilsMessageSeverityFlagsEXT,
        r#type: vk::DebugUtilsMessageTypeFlagsEXT,
        msg: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> vk::Bool32 {
        unsafe {
            match flag {
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                    log::trace!("{:?} - {:?}", r#type, CStr::from_ptr((*msg).p_message));
                }
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                    log::info!("{:?} - {:?}", r#type, CStr::from_ptr((*msg).p_message));
                }
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                    log::warn!("{:?} - {:?}", r#type, CStr::from_ptr((*msg).p_message));
                }
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                    log::error!("{:?} - {:?}", r#type, CStr::from_ptr((*msg).p_message));
                }
                _ => {
                    unreachable!("flag: {:?}", flag);
                }
            }
        }
        vk::FALSE
    }

    pub fn run(&mut self) {
        log::info!("Running application");
    }
}
impl Drop for App {
    fn drop(&mut self) {
        log::info!("Dropping application");
        unsafe { self.instance.destroy_instance(None) };
    }
}
