fn main() {
    #[cfg(any(feature = "gui", feature = "gui-readback"))]
    {
        println!("cargo:rerun-if-changed=ui");
        slint_build::compile("ui/main.slint").expect("failed to compile Slint UI");
    }
}
