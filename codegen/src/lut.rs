use std::{fs::File, io::Write};

pub fn build() {
    use std::path::Path;

    let out_dir = ::std::env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lut.rs");
    let mut writer = File::create(dest_path).expect("couldn't create lut.rs");
    build_transfer_fn(&mut writer);
}

type TransferFn = Box<dyn Fn(f64) -> f64>;

struct LutEntryU8 {
    fn_type: String,
    fn_type_uppercase: String,
    from_linear: TransferFn,
    into_linear: TransferFn,
}

#[cfg(feature = "prophoto_lut")]
struct LutEntryU16 {
    fn_type: String,
    fn_type_uppercase: String,
    from_linear: TransferFn,
    into_linear: TransferFn,
    is_linear_as_until: Option<(f64, f64)>,
}

impl LutEntryU8 {
    fn new<F1: Fn(f64) -> f64 + 'static, F2: Fn(f64) -> f64 + 'static>(
        fn_type: &str,
        fn_type_uppercase: &str,
        from_linear: F1,
        into_linear: F2,
    ) -> Self {
        Self {
            fn_type: fn_type.to_owned(),
            fn_type_uppercase: fn_type_uppercase.to_owned(),
            from_linear: Box::new(from_linear),
            into_linear: Box::new(into_linear),
        }
    }
}

#[cfg(feature = "prophoto_lut")]
impl LutEntryU16 {
    fn new<F1: Fn(f64) -> f64 + 'static, F2: Fn(f64) -> f64 + 'static>(
        fn_type: &str,
        fn_type_uppercase: &str,
        from_linear: F1,
        into_linear: F2,
    ) -> Self {
        Self {
            fn_type: fn_type.to_owned(),
            fn_type_uppercase: fn_type_uppercase.to_owned(),
            from_linear: Box::new(from_linear),
            into_linear: Box::new(into_linear),
            is_linear_as_until: None,
        }
    }

    fn new_with_linear<F1: Fn(f64) -> f64 + 'static, F2: Fn(f64) -> f64 + 'static>(
        fn_type: &str,
        fn_type_uppercase: &str,
        from_linear: F1,
        into_linear: F2,
        linear_scale: f64,
        linear_end: f64,
    ) -> Self {
        let mut entry = Self::new(fn_type, fn_type_uppercase, from_linear, into_linear);
        entry.is_linear_as_until = Some((linear_scale, linear_end));
        entry
    }
}

pub fn build_transfer_fn(writer: &mut File) {
    let entries_u8: Vec<LutEntryU8> = vec![
        LutEntryU8::new(
            "Srgb",
            "SRGB",
            |linear| {
                if linear <= 0.0031308 {
                    12.92 * linear
                } else {
                    linear.powf(1.0 / 2.4).mul_add(1.055, -0.055)
                }
            },
            |encoded| {
                if encoded <= 0.04045 {
                    encoded / 12.92
                } else {
                    ((encoded + 0.055) / 1.055).powf(2.4)
                }
            },
        ),
        #[cfg(feature = "rec_oetf_lut")]
        LutEntryU8::new(
            "RecOetf",
            "REC_OETF",
            |linear| {
                const ALPHA: f64 = 1.09929682680944;
                const BETA: f64 = 0.018053968510807;
                if linear < BETA {
                    4.5 * linear
                } else {
                    linear.powf(0.45).mul_add(ALPHA, 1.0 - ALPHA)
                }
            },
            |encoded| {
                const ALPHA: f64 = 1.09929682680944;
                const BETA: f64 = 0.018053968510807;
                if encoded < 4.5 * BETA {
                    encoded / 4.5
                } else {
                    ((encoded + ALPHA - 1.0) / ALPHA).powf(1.0 / 0.45)
                }
            },
        ),
        #[cfg(feature = "adobe_rgb_lut")]
        LutEntryU8::new(
            "AdobeRgb",
            "ADOBE_RGB",
            |linear| linear.powf(256.0 / 563.0),
            |encoded| encoded.powf(563.0 / 256.0),
        ),
        #[cfg(feature = "p3_gamma_lut")]
        LutEntryU8::new(
            "P3Gamma",
            "P3_GAMMA",
            |linear| linear.powf(1.0 / 2.6),
            |encoded| encoded.powf(2.6),
        ),
    ];
    #[cfg(feature = "prophoto_lut")]
    let entries_u16: Vec<LutEntryU16> = vec![
        #[cfg(feature = "prophoto_lut")]
        LutEntryU16::new_with_linear(
            "ProPhotoRgb",
            "PROPHOTO_RGB",
            |linear| {
                if linear < 0.001953125 {
                    16.0 * linear
                } else {
                    linear.powf(1.0 / 1.8)
                }
            },
            |encoded| {
                if encoded < 0.03125 {
                    encoded / 16.0
                } else {
                    encoded.powf(1.8)
                }
            },
            16.0,
            0.001953125,
        ),
    ];

    write!(writer, "use crate::encoding::{{FromLinear, IntoLinear").unwrap();
    for entry in &entries_u8 {
        write!(writer, ", {}", entry.fn_type).unwrap();
    }
    #[cfg(feature = "prophoto_lut")]
    for entry in &entries_u16 {
        write!(writer, ", {}", entry.fn_type).unwrap();
    }
    writeln!(writer, "}};").unwrap();

    gen_into_linear_lut_u8(writer, &entries_u8);
    #[cfg(feature = "prophoto_lut")]
    gen_into_linear_lut_u16(writer, &entries_u16);
    gen_from_linear_lut_u8(writer, &entries_u8);
    #[cfg(feature = "prophoto_lut")]
    gen_from_linear_lut_u16(writer, &entries_u16)
}

fn gen_into_linear_lut_u8(writer: &mut File, entries: &[LutEntryU8]) {
    for LutEntryU8 {
        fn_type,
        fn_type_uppercase,
        into_linear,
        ..
    } in entries
    {
        let table_size = 1 << 8;
        let mut table = Vec::new();
        for i in 0..table_size {
            let encoded = (i as f64) / ((table_size - 1) as f64);
            let linear = into_linear(encoded);
            // Handle integer floats printing without decimal
            let float_string = if linear <= 0.0 {
                "\t0.0,\n".to_owned()
            } else if linear >= 1.0 {
                "\t1.0,\n".to_owned()
            } else {
                format!("\t{linear},\n")
            };
            table.extend_from_slice(float_string.as_bytes());
        }

        let table_name = format!("{fn_type_uppercase}_U8_TO_F64");
        writeln!(writer, "const {table_name}: [f64; {table_size}] = [").unwrap();
        writer.write_all(&table).unwrap();
        writeln!(writer, "];").unwrap();

        writeln!(
            writer,
            "impl IntoLinear<f64, u8> for {fn_type} {{\
        \n\t#[inline]\
        \n\tfn into_linear(encoded: u8) -> f64 {{\
        \n\t\t{table_name}[encoded as usize]\
        \n\t}}\
        \n}}"
        )
        .unwrap();

        writeln!(
            writer,
            "impl IntoLinear<f32, u8> for {fn_type} {{\
        \n\t#[inline]\
        \n\tfn into_linear(encoded: u8) -> f32 {{\
        \n\t\t{table_name}[encoded as usize] as f32\
        \n\t}}\
        \n}}"
        )
        .unwrap();
    }
}

#[cfg(feature = "prophoto_lut")]
fn gen_into_linear_lut_u16(writer: &mut File, entries: &[LutEntryU16]) {
    for LutEntryU16 {
        fn_type,
        fn_type_uppercase,
        into_linear,
        ..
    } in entries
    {
        let table_size = 1 << 16;
        let mut table = Vec::new();
        for i in 0..table_size {
            let encoded = (i as f64) / ((table_size - 1) as f64);
            let linear = into_linear(encoded);
            // Handle integer floats printing without decimal
            let float_string = if linear <= 0.0 {
                "\t0.0,\n".to_owned()
            } else if linear >= 1.0 {
                "\t1.0,\n".to_owned()
            } else {
                format!("\t{linear},\n")
            };
            table.extend_from_slice(float_string.as_bytes());
        }

        let table_name = format!("{fn_type_uppercase}_U16_TO_F64");
        writeln!(writer, "static {table_name}: [f64; {table_size}] = [").unwrap();
        writer.write_all(&table).unwrap();
        writeln!(writer, "];").unwrap();

        writeln!(
            writer,
            "impl IntoLinear<f64, u16> for {fn_type} {{\
            \n\t#[inline]\
            \n\tfn into_linear(encoded: u16) -> f64 {{\
            \n\t\t{table_name}[encoded as usize]\
            \n\t}}\
            \n}}"
        )
        .unwrap();

        writeln!(
            writer,
            "impl IntoLinear<f32, u16> for {fn_type} {{\
            \n\t#[inline]\
            \n\tfn into_linear(encoded: u16) -> f32 {{\
            \n\t\t{table_name}[encoded as usize] as f32\
            \n\t}}\
            \n}}"
        )
        .unwrap();
    }
}

fn gen_from_linear_lut_u8(writer: &mut File, entries: &[LutEntryU8]) {
    for LutEntryU8 {
        fn_type,
        fn_type_uppercase,
        from_linear,
        into_linear,
    } in entries
    {
        let max_float_bits: u32 = 0x3f7fffff; // 1.0 - f32::EPSILON
        let min_float_bits: u32 = (into_linear(0.5 / 255.0) as f32).to_bits() & 0xff800000; // Any input less than this maps to 0
        let exp_table_size = ((max_float_bits - min_float_bits) >> 23) + 1;
        'man_bits: for man_index_size in 0..=15 {
            let table_size = exp_table_size << man_index_size;
            let bucket_index_size = 23 - man_index_size;
            let bucket_size = 1 << bucket_index_size;
            let mut table = Vec::new();
            for i in 0..table_size {
                let start = min_float_bits + (i << bucket_index_size);

                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_x2 = 0.0;
                for j in 0..bucket_size {
                    let x = (j >> (bucket_index_size - 8)) as f64;
                    let linear: f64 = f32::from_bits(start + j) as f64;
                    let y = 255.0 * from_linear(linear) + 0.5;
                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_x2 += x * x;
                }
                let scale = (bucket_size as f64 * sum_xy - sum_x * sum_y)
                    / (bucket_size as f64 * sum_x2 - sum_x * sum_x);
                let bias = (sum_y - scale * sum_x) / bucket_size as f64;
                let scale_uint = (scale * 65536.0 + 0.5) as u32;
                let bias_uint = ((bias * 128.0 + 0.5) as u32) << 9;

                for j in 0..bucket_size {
                    let x = j >> (bucket_index_size - 8);
                    let linear = f32::from_bits(start + j) as f64;
                    let y = 255.0 * from_linear(linear);
                    let y_approx = (scale_uint * x + bias_uint) >> 16;
                    if (y - y_approx as f64).abs() > 0.6 {
                        continue 'man_bits;
                    }
                }

                if i % 8 == 0 {
                    table.extend_from_slice("\t".as_bytes());
                }
                table.extend_from_slice(
                    format!("{:#010x}, ", (bias_uint << 7) | scale_uint).as_bytes(),
                );
                if i % 8 == 7 {
                    table.extend_from_slice("\n".as_bytes());
                }
            }

            let table_name = format!("TO_{fn_type_uppercase}_U8");
            writeln!(writer, "const {table_name}: [u32; {table_size}] = [").unwrap();
            writer.write_all(&table).unwrap();
            writeln!(writer, "\n];").unwrap();

            let entry_shift = 23 - man_index_size;
            let man_shift = 15 - man_index_size;

            writeln!(
                writer,
                "impl FromLinear<f64, u8> for {fn_type} {{\
                \n\t#[inline]\
                \n\tfn from_linear(linear: f64) -> u8 {{\
                \n\t\t{fn_type}::from_linear(linear as f32)\
                \n\t}}\
                \n}}"
            )
            .unwrap();

            let min_float_string = format!("{:#x}", min_float_bits);
            writeln!(
                writer,
                "impl FromLinear<f32, u8> for {fn_type} {{\
                    \n\t#[inline]\
                    \n\tfn from_linear(linear: f32) -> u8 {{\
                        \n\t\tconst MAX_FLOAT_BITS: u32 = 0x3f7fffff; // 1.0 - f32::EPSILON\
                        \n\t\tconst MIN_FLOAT_BITS: u32 = {min_float_string}; // 2^(-{exp_table_size})\
                        \n\t\tlet max_float = f32::from_bits(MAX_FLOAT_BITS);\
                            \n\t\tlet min_float = f32::from_bits(MIN_FLOAT_BITS);\
                            \n\n\t\tlet mut input = linear;\
                        \n\t\tif input.partial_cmp(&min_float) != Some(core::cmp::Ordering::Greater) {{\
                            \n\t\t\tinput = min_float;\
                        \n\t\t}} else if input > max_float {{\
                            \n\t\t\tinput = max_float;\
                        \n\t\t}}
                        \n\t\tlet input_bits = input.to_bits();\
                        \n\t\t#[cfg(test)]\
                        \n\t\t{{\
                            \n\t\t\tdebug_assert!((MIN_FLOAT_BITS..=MAX_FLOAT_BITS).contains(&input_bits));\
                        \n\t\t}}\
                        \n\n\t\tlet entry = {{\
                            \n\t\t\tlet i = ((input_bits - MIN_FLOAT_BITS) >> {entry_shift}) as usize;\
                            \n\t\t\t#[cfg(test)]\
                            \n\t\t\t{{\
                                \n\t\t\t\tdebug_assert!({table_name}.get(i).is_some());\
                            \n\t\t\t}}\
                            \n\t\t\tunsafe {{ *{table_name}.get_unchecked(i) }}\
                        \n\t\t}};\
                        \n\t\tlet bias = (entry >> 16) << 9;\
                        \n\t\tlet scale = entry & 0xffff;\
                        \n\n\t\tlet t = (input_bits >> {man_shift}) & 0xff;\
                        \n\t\tlet res = (bias + scale * t) >> 16;\
                        \n\t\t#[cfg(test)]\
                        \n\t\t{{\
                            \n\t\t\tdebug_assert!(res < 256, \"{{}}\", res);\
                        \n\t\t}}\
                        \n\t\tres as u8\
                    \n\t}}\
                \n}}"
            )
            .unwrap();
            break;
        }
    }
}

#[cfg(feature = "prophoto_lut")]
fn gen_from_linear_lut_u16(writer: &mut File, entries: &[LutEntryU16]) {
    for LutEntryU16 {
        fn_type,
        fn_type_uppercase,
        from_linear,
        into_linear,
        is_linear_as_until,
    } in entries
    {
        let max_float_bits: u32 = 0x3f7fffff; // 1.0 - f32::EPSILON
        let min_float_bits: u32 = is_linear_as_until
            .map(|(_, linear_end)| linear_end as f32)
            .unwrap_or_else(|| into_linear(0.5 / 65535.0) as f32)
            .to_bits()
            & 0xff800000;
        let exp_table_size = ((max_float_bits - min_float_bits) >> 23) + 1;
        'man_bits: for man_index_size in 0..=7 {
            let table_size = exp_table_size << man_index_size;
            let bucket_index_size = 23 - man_index_size;
            let bucket_size = 1 << bucket_index_size;
            let mut table = Vec::new();
            for i in 0..table_size {
                let start = min_float_bits + (i << bucket_index_size);

                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_x2 = 0.0;
                for j in 0..bucket_size {
                    let x = (j >> (bucket_index_size - 16)) as f64;
                    let linear: f64 = f32::from_bits(start + j) as f64;
                    let y = 65535.0 * from_linear(linear) + 0.5;
                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_x2 += x * x;
                }
                let scale = (bucket_size as f64 * sum_xy - sum_x * sum_y)
                    / (bucket_size as f64 * sum_x2 - sum_x * sum_x);
                let bias = (sum_y - scale * sum_x) / bucket_size as f64;
                let scale_uint = (scale * 2.0f64.powi(32) + 0.5) as u64;
                let bias_uint = ((bias * 2.0f64.powi(15) + 0.5) as u64) << 17;

                for j in 0..bucket_size {
                    let x = j >> (bucket_index_size - 16);
                    let linear = f32::from_bits(start + j) as f64;
                    let y = 65535.0 * from_linear(linear);
                    let y_approx = (scale_uint * x as u64 + bias_uint) >> 32;
                    if (y - y_approx as f64).abs() > 0.6 {
                        continue 'man_bits;
                    }
                }

                if i % 4 == 0 {
                    table.extend_from_slice("\t".as_bytes());
                }
                table.extend_from_slice(
                    format!("{:#018x}, ", (bias_uint << 15) | scale_uint).as_bytes(),
                );
                if i % 4 == 3 {
                    table.extend_from_slice("\n".as_bytes());
                }
            }

            let table_name = format!("TO_{fn_type_uppercase}_U16");
            writeln!(writer, "const {table_name}: [u64; {table_size}] = [").unwrap();
            writer.write_all(&table).unwrap();
            writeln!(writer, "\n];").unwrap();

            let entry_shift = 23 - man_index_size;
            let man_shift = 7 - man_index_size;

            writeln!(
                writer,
                "impl FromLinear<f64, u16> for {fn_type} {{\
                \n\t#[inline]\
                \n\tfn from_linear(linear: f64) -> u16 {{\
                \n\t\t{fn_type}::from_linear(linear as f32)\
                \n\t}}\
                \n}}"
            )
            .unwrap();

            let min_float_string = format!("{:#x}", min_float_bits);
            writeln!(
                writer,
                "impl FromLinear<f32, u16> for {fn_type} {{\
                    \n\t#[inline]\
                    \n\tfn from_linear(linear: f32) -> u16 {{\
                        \n\t\tconst MAX_FLOAT_BITS: u32 = 0x3f7fffff; // 1.0 - f32::EPSILON\
                        \n\t\tconst MIN_FLOAT_BITS: u32 = {min_float_string}; // 2^(-{exp_table_size})\
                        \n\t\tlet max_float = f32::from_bits(MAX_FLOAT_BITS);\
                            \n\t\tlet min_float = f32::from_bits(MIN_FLOAT_BITS);\
                            \n\n\t\tlet mut input = linear;"
                )
                .unwrap();
            writeln!(
                writer,
                "\
                            \t\tif input.partial_cmp(&{0}) != Some(core::cmp::Ordering::Greater) {{\
                                \n\t\t\tinput = {0};",
                if is_linear_as_until.is_some() {
                    "0.0"
                } else {
                    "min_float"
                }
            )
            .unwrap();
            writeln!(
                writer,
                "\
                            \t\t}} else if input > max_float {{\
                            \n\t\t\tinput = max_float;\
                            \n\t\t}}"
            )
            .unwrap();
            if let Some((linear_scale, _)) = is_linear_as_until {
                let scale = linear_scale * 65535.0;
                let magic_value = f32::from_bits((127 + 23) << 23);
                writeln!(
                        writer,
                        "\
                            \t\tif input < min_float {{\
                                \n\t\t\treturn (({scale}f32 * input + {magic_value}f32).to_bits() & 65535) as u16;\
                            \n\t\t}}"
                    ).unwrap();
            }
            writeln!(
                    writer,
                    "\
                        \n\t\tlet input_bits = input.to_bits();\
                        \n\t\t#[cfg(test)]\
                        \n\t\t{{\
                            \n\t\t\tdebug_assert!((MIN_FLOAT_BITS..=MAX_FLOAT_BITS).contains(&input_bits));\
                        \n\t\t}}\
                        \n\n\t\tlet entry = {{\
                            \n\t\t\tlet i = ((input_bits - MIN_FLOAT_BITS) >> {entry_shift}) as usize;\
                            \n\t\t\t#[cfg(test)]\
                            \n\t\t\t{{\
                                \n\t\t\t\tdebug_assert!({table_name}.get(i).is_some());\
                            \n\t\t\t}}\
                            \n\t\t\tunsafe {{ *{table_name}.get_unchecked(i) }}\
                        \n\t\t}};\
                        \n\t\tlet bias = (entry >> 32) << 17;\
                        \n\t\tlet scale = entry & 0xffff_ffff;"
            ).unwrap();
            if man_shift == 0 {
                writeln!(writer, "\n\t\tlet t = input_bits as u64 & 0xffff;").unwrap();
            } else {
                writeln!(
                    writer,
                    "\n\t\tlet t = (input_bits as u64 >> {man_shift}) & 0xffff;"
                )
                .unwrap();
            }
            writeln!(
                writer,
                "\
                        \t\tlet res = (bias + scale * t) >> 32;\
                        \n\t\t#[cfg(test)]\
                        \n\t\t{{\
                            \n\t\t\tdebug_assert!(res < 65536, \"{{}}\", res);\
                        \n\t\t}}\
                        \n\t\tres as u16\
                    \n\t}}\
                \n}}"
            )
            .unwrap();
            break;
        }
    }
}
