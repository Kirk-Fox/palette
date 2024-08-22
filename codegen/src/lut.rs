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
    into_linear: TransferFn,
    linear_scale: Option<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

#[cfg(feature = "prophoto_lut")]
struct LutEntryU16 {
    fn_type: String,
    fn_type_uppercase: String,
    into_linear: TransferFn,
    linear_scale: Option<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl LutEntryU8 {
    fn new(
        fn_type: &str,
        fn_type_uppercase: &str,
        is_linear_as_until: Option<(f64, f64)>,
        gamma: f64,
    ) -> Self {
        let (linear_scale, alpha, beta) =
            if let Some((linear_scale, linear_end)) = is_linear_as_until {
                (
                    Some(linear_scale),
                    (linear_scale * linear_end - 1.0) / (linear_end.powf(gamma.recip()) - 1.0),
                    linear_end,
                )
            } else {
                (None, 1.0, 0.0)
            };
        Self {
            fn_type: fn_type.to_owned(),
            fn_type_uppercase: fn_type_uppercase.to_owned(),
            into_linear: Box::new(move |encoded| match linear_scale {
                Some(scale) if encoded <= scale * beta => encoded / scale,
                _ => ((encoded + alpha - 1.0) / alpha).powf(gamma),
            }),
            linear_scale,
            alpha,
            beta,
            gamma,
        }
    }
}

#[cfg(feature = "prophoto_lut")]
impl LutEntryU16 {
    fn new(
        fn_type: &str,
        fn_type_uppercase: &str,
        is_linear_as_until: Option<(f64, f64)>,
        gamma: f64,
    ) -> Self {
        let (linear_scale, alpha, beta) =
            if let Some((linear_scale, linear_end)) = is_linear_as_until {
                (
                    Some(linear_scale),
                    (linear_scale * linear_end - 1.0) / (linear_end.powf(gamma.recip()) - 1.0),
                    linear_end,
                )
            } else {
                (None, 1.0, 0.0)
            };
        Self {
            fn_type: fn_type.to_owned(),
            fn_type_uppercase: fn_type_uppercase.to_owned(),
            into_linear: Box::new(move |encoded| match linear_scale {
                Some(scale) if encoded <= scale * beta => encoded / scale,
                _ => ((encoded + alpha - 1.0) / alpha).powf(gamma),
            }),
            linear_scale,
            alpha,
            beta,
            gamma,
        }
    }
}

pub fn build_transfer_fn(writer: &mut File) {
    let entries_u8: Vec<LutEntryU8> = vec![
        LutEntryU8::new("Srgb", "SRGB", Some((12.92, 0.0031308)), 2.4),
        #[cfg(feature = "rec_oetf_lut")]
        LutEntryU8::new(
            "RecOetf",
            "REC_OETF",
            Some((4.5, 0.018053968510807)),
            1.0 / 0.45,
        ),
        #[cfg(feature = "adobe_rgb_lut")]
        LutEntryU8::new("AdobeRgb", "ADOBE_RGB", None, 563.0 / 256.0),
        #[cfg(feature = "p3_gamma_lut")]
        LutEntryU8::new("P3Gamma", "P3_GAMMA", None, 2.6),
    ];
    #[cfg(feature = "prophoto_lut")]
    let entries_u16: Vec<LutEntryU16> = vec![
        #[cfg(feature = "prophoto_lut")]
        LutEntryU16::new(
            "ProPhotoRgb",
            "PROPHOTO_RGB",
            Some((16.0, 0.001953125)),
            1.8,
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
    gen_from_linear_lut_u16(writer, &entries_u16);
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

fn integrate_linear(
    (start_x, start_t): (f64, f64),
    (end_x, end_t): (f64, f64),
    linear_scale: f64,
    exp_scale: f64,
) -> (f64, f64) {
    let antiderive_y = |x: f64| 0.5 * linear_scale * exp_scale * x * x;
    let antiderive_ty =
        |x: f64, t: f64| 0.5 * linear_scale * exp_scale * x * x * (t - exp_scale * x / 3.0);

    (
        antiderive_y(end_x) - antiderive_y(start_x),
        antiderive_ty(end_x, end_t) - antiderive_ty(start_x, start_t),
    )
}

fn integrate_exponential(
    (start_x, start_t): (f64, f64),
    (end_x, end_t): (f64, f64),
    alpha: f64,
    gamma: f64,
    exp_scale: f64,
) -> (f64, f64) {
    let antiderive_y = |x: f64, t: f64| {
        alpha * gamma * exp_scale * x * x.powf(gamma.recip()) / (1.0 + gamma) + (1.0 - alpha) * t
    };
    let antiderive_ty = |x: f64, t: f64| {
        alpha
            * gamma
            * exp_scale
            * x
            * x.powf(gamma.recip())
            * (t - gamma * exp_scale * x / (1.0 + 2.0 * gamma))
            / (1.0 + gamma)
            + 0.5 * (1.0 - alpha) * t * t
    };

    (
        antiderive_y(end_x, end_t) - antiderive_y(start_x, start_t),
        antiderive_ty(end_x, end_t) - antiderive_ty(start_x, start_t),
    )
}

fn gen_from_linear_lut_u8(writer: &mut File, entries: &[LutEntryU8]) {
    for LutEntryU8 {
        fn_type,
        fn_type_uppercase,
        into_linear,
        linear_scale,
        alpha,
        beta,
        gamma,
        ..
    } in entries
    {
        // 1.0 - f32::EPSILON
        let max_float_bits: u32 = 0x3f7fffff;
        // Any input less than or equal to this maps to 0
        let min_float_bits: u32 = ((into_linear(0.5 / 255.0) as f32).to_bits() - 1) & 0xff800000;

        let exp_table_size = ((max_float_bits - min_float_bits) >> 23) + 1;
        let man_index_size = 3;
        let table_size = exp_table_size << man_index_size;
        let bucket_index_size = 23 - man_index_size;
        let bucket_size = 1 << bucket_index_size;
        let mut table = Vec::new();

        for i in 0..table_size {
            let start = min_float_bits + (i << bucket_index_size);
            let end = start + bucket_size;
            let start_float = f32::from_bits(start) as f64;
            let end_float = f32::from_bits(end) as f64;

            let beta_bits = (*beta as f32).to_bits();
            let exp_scale = 2.0f64.powi(158 - ((start >> 23) as i32) - bucket_index_size);

            let (integral_y, integral_ty) = match linear_scale {
                Some(scale) if end <= beta_bits => {
                    integrate_linear((start_float, 0.0), (end_float, 256.0), *scale, exp_scale)
                }
                Some(scale) if start < beta_bits => {
                    let beta_t =
                        (beta_bits & (bucket_size - 1)) as f64 * 2.0f64.powi(8 - bucket_index_size);
                    let integral_linear =
                        integrate_linear((start_float, 0.0), (*beta, beta_t), *scale, exp_scale);
                    let integral_exponential = integrate_exponential(
                        (*beta, beta_t),
                        (end_float, 256.0),
                        *alpha,
                        *gamma,
                        exp_scale,
                    );
                    (
                        integral_linear.0 + integral_exponential.0,
                        integral_linear.1 + integral_exponential.1,
                    )
                }
                _ => integrate_exponential(
                    (start_float, 0.0),
                    (end_float, 256.0),
                    *alpha,
                    *gamma,
                    exp_scale,
                ),
            };

            const INTEGRAL_T: f64 = 32768.0;
            const INTEGRAL_T2: f64 = 16777216.0 / 3.0;

            let scale = (256.0 * integral_ty - INTEGRAL_T * integral_y)
                / (256.0 * INTEGRAL_T2 - INTEGRAL_T * INTEGRAL_T);
            let bias = (integral_y - scale * INTEGRAL_T) / 256.0;
            let scale_uint = (255.0 * scale * 65536.0 + 0.5) as u32;
            let bias_uint = (((255.0 * bias + 0.5) * 128.0 + 0.5) as u32) << 9;

            if i % 8 == 0 {
                table.extend_from_slice("\t".as_bytes());
            }
            table
                .extend_from_slice(format!("{:#010x}, ", (bias_uint << 7) | scale_uint).as_bytes());
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
    }
}

#[cfg(feature = "prophoto_lut")]
fn gen_from_linear_lut_u16(writer: &mut File, entries: &[LutEntryU16]) {
    for LutEntryU16 {
        fn_type,
        fn_type_uppercase,
        into_linear,
        linear_scale,
        alpha,
        beta,
        gamma,
        ..
    } in entries
    {
        // 1.0 - f32::EPSILON
        let max_float_bits: u32 = 0x3f7fffff;
        let min_float_bits: u32 = (*beta as f32)
            .to_bits()
            .max((into_linear(0.5 / 65535.0) as f32).to_bits() - 1)
            & 0xff800000;

        let exp_table_size = ((max_float_bits - min_float_bits) >> 23) + 1;
        let man_index_size = 7;
        let table_size = exp_table_size << man_index_size;
        let bucket_index_size = 23 - man_index_size;
        let bucket_size = 1 << bucket_index_size;
        let mut table = Vec::new();

        for i in 0..table_size {
            let start = min_float_bits + (i << bucket_index_size);
            let end = start + bucket_size;
            let start_float = f32::from_bits(start) as f64;
            let end_float = f32::from_bits(end) as f64;

            let beta_bits = (*beta as f32).to_bits();
            let exp_scale = 2.0f64.powi(166 - ((start >> 23) as i32) - bucket_index_size);

            let (integral_y, integral_ty) = match linear_scale {
                Some(scale) if end <= beta_bits => {
                    integrate_linear((start_float, 0.0), (end_float, 65536.0), *scale, exp_scale)
                }
                Some(scale) if start < beta_bits => {
                    let beta_t = (beta_bits & (bucket_size - 1)) as f64
                        * 2.0f64.powi(16 - bucket_index_size);
                    let integral_linear =
                        integrate_linear((start_float, 0.0), (*beta, beta_t), *scale, exp_scale);
                    let integral_exponential = integrate_exponential(
                        (*beta, beta_t),
                        (end_float, 65536.0),
                        *alpha,
                        *gamma,
                        exp_scale,
                    );
                    (
                        integral_linear.0 + integral_exponential.0,
                        integral_linear.1 + integral_exponential.1,
                    )
                }
                _ => integrate_exponential(
                    (start_float, 0.0),
                    (end_float, 65536.0),
                    *alpha,
                    *gamma,
                    exp_scale,
                ),
            };

            const INTEGRAL_T: f64 = 2147483648.0;
            const INTEGRAL_T2: f64 = 281474976710656.0 / 3.0;

            let scale = (65536.0 * integral_ty - INTEGRAL_T * integral_y)
                / (65536.0 * INTEGRAL_T2 - INTEGRAL_T * INTEGRAL_T);
            let bias = (integral_y - scale * INTEGRAL_T) / 65536.0;
            let scale_uint = (65535.0 * scale * 4294967296.0 + 0.5) as u64;
            let bias_uint = (((65535.0 * bias + 0.5) * 32768.0 + 0.5) as u64) << 17;

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
            if linear_scale.is_some() {
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
        if let Some(scale) = linear_scale {
            let adj_scale = scale * 65535.0;
            let magic_value = f32::from_bits((127 + 23) << 23);
            writeln!(
                    writer,
                    "\
                        \t\tif input < min_float {{\
                            \n\t\t\treturn (({adj_scale}f32 * input + {magic_value}f32).to_bits() & 65535) as u16;\
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
    }
}
