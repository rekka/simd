#![feature(test)]
#![feature(cfg_target_feature)]
extern crate test;
extern crate simd;

use test::black_box as bb;
use test::Bencher as B;
use simd::f32x4;
#[cfg(target_feature = "avx")]
use simd::x86::avx::{f32x8, f64x4};
// #[cfg(target_feature = "avx2")]
// use simd::x86::avx2::Avx2F32x8;


#[bench]
fn multiply_naive(b: &mut B) {
    let x = [[1.0_f32; 4]; 4];
    let y = [[2.0; 4]; 4];
    b.iter(|| {
        for _ in 0..100 {
        let (x, y) = bb((&x, &y));

        bb(&[[x[0][0] * y[0][0] + x[1][0] * y[0][1] + x[2][0] * y[0][2] + x[3][0] * y[0][3],
            x[0][1] * y[0][0] + x[1][1] * y[0][1] + x[2][1] * y[0][2] + x[3][1] * y[0][3],
            x[0][2] * y[0][0] + x[1][2] * y[0][1] + x[2][2] * y[0][2] + x[3][2] * y[0][3],
            x[0][3] * y[0][0] + x[1][3] * y[0][1] + x[2][3] * y[0][2] + x[3][3] * y[0][3]],
           [x[0][0] * y[1][0] + x[1][0] * y[1][1] + x[2][0] * y[1][2] + x[3][0] * y[1][3],
            x[0][1] * y[1][0] + x[1][1] * y[1][1] + x[2][1] * y[1][2] + x[3][1] * y[1][3],
            x[0][2] * y[1][0] + x[1][2] * y[1][1] + x[2][2] * y[1][2] + x[3][2] * y[1][3],
            x[0][3] * y[1][0] + x[1][3] * y[1][1] + x[2][3] * y[1][2] + x[3][3] * y[1][3]],
           [x[0][0] * y[2][0] + x[1][0] * y[2][1] + x[2][0] * y[2][2] + x[3][0] * y[2][3],
            x[0][1] * y[2][0] + x[1][1] * y[2][1] + x[2][1] * y[2][2] + x[3][1] * y[2][3],
            x[0][2] * y[2][0] + x[1][2] * y[2][1] + x[2][2] * y[2][2] + x[3][2] * y[2][3],
            x[0][3] * y[2][0] + x[1][3] * y[2][1] + x[2][3] * y[2][2] + x[3][3] * y[2][3]],
           [x[0][0] * y[3][0] + x[1][0] * y[3][1] + x[2][0] * y[3][2] + x[3][0] * y[3][3],
            x[0][1] * y[3][0] + x[1][1] * y[3][1] + x[2][1] * y[3][2] + x[3][1] * y[3][3],
            x[0][2] * y[3][0] + x[1][2] * y[3][1] + x[2][2] * y[3][2] + x[3][2] * y[3][3],
            x[0][3] * y[3][0] + x[1][3] * y[3][1] + x[2][3] * y[3][2] + x[3][3] * y[3][3]],
             ]);
        }
    })
}

#[bench]
fn multiply_simd4_32(b: &mut B) {
    let x = [f32x4::splat(1.0_f32); 4];
    let y = [f32x4::splat(2.0); 4];
    b.iter(|| {
        for _ in 0..100 {
        let (x, y) = bb((&x, &y));

        let y0 = y[0];
        let y1 = y[1];
        let y2 = y[2];
        let y3 = y[3];
        bb(&[f32x4::splat(y0.extract(0)) * x[0] +
             f32x4::splat(y0.extract(1)) * x[1] +
             f32x4::splat(y0.extract(2)) * x[2] +
             f32x4::splat(y0.extract(3)) * x[3],
             f32x4::splat(y1.extract(0)) * x[0] +
             f32x4::splat(y1.extract(1)) * x[1] +
             f32x4::splat(y1.extract(2)) * x[2] +
             f32x4::splat(y1.extract(3)) * x[3],
             f32x4::splat(y2.extract(0)) * x[0] +
             f32x4::splat(y2.extract(1)) * x[1] +
             f32x4::splat(y2.extract(2)) * x[2] +
             f32x4::splat(y2.extract(3)) * x[3],
             f32x4::splat(y3.extract(0)) * x[0] +
             f32x4::splat(y3.extract(1)) * x[1] +
             f32x4::splat(y3.extract(2)) * x[2] +
             f32x4::splat(y3.extract(3)) * x[3],
             ]);
        }
    })
}

#[cfg(target_feature = "avx")]
#[bench]
fn multiply_simd8_32(b: &mut B) {
    let x = [f32x8::splat(1.0_f32); 2];
    let y = [f32x8::splat(2.0); 2];

    fn splat2(low: f32, high: f32) -> f32x8 {
        f32x8::new(low, low, low, low, high, high, high, high)
    }

    fn swap_lanes(x: f32x8) -> f32x8 {
        f32x8::new(x.extract(4), x.extract(5), x.extract(6), x.extract(7),
            x.extract(0), x.extract(1), x.extract(2), x.extract(3))
    }

    b.iter(|| {
        for _ in 0..100 {
        let (x, y) = bb((&x, &y));

        let x01 = x[0];
        let x10 = swap_lanes(x01);
        let x23 = x[1];
        let x32 = swap_lanes(x23);

        let y01 = y[0];
        let y23 = y[1];
        bb(&[splat2(y01.extract(0), y01.extract(5)) * x01 +
             splat2(y01.extract(2), y01.extract(7)) * x23 +
             splat2(y01.extract(1), y01.extract(4)) * x10 +
             splat2(y01.extract(3), y01.extract(6)) * x32,

             splat2(y23.extract(0), y23.extract(5)) * x01 +
             splat2(y23.extract(2), y23.extract(7)) * x23 +
             splat2(y23.extract(1), y23.extract(4)) * x10 +
             splat2(y23.extract(3), y23.extract(6)) * x32,
             ]);
        }
    })
}

#[cfg(target_feature = "avx")]
#[bench]
fn multiply_simd4_64(b: &mut B) {
    let x = [f64x4::splat(1.0_f64); 4];
    let y = [f64x4::splat(2.0); 4];
    b.iter(|| {
        for _ in 0..100 {
        let (x, y) = bb((&x, &y));

        let y0 = y[0];
        let y1 = y[1];
        let y2 = y[2];
        let y3 = y[3];
        bb(&[f64x4::splat(y0.extract(0)) * x[0] +
             f64x4::splat(y0.extract(1)) * x[1] +
             f64x4::splat(y0.extract(2)) * x[2] +
             f64x4::splat(y0.extract(3)) * x[3],
             f64x4::splat(y1.extract(0)) * x[0] +
             f64x4::splat(y1.extract(1)) * x[1] +
             f64x4::splat(y1.extract(2)) * x[2] +
             f64x4::splat(y1.extract(3)) * x[3],
             f64x4::splat(y2.extract(0)) * x[0] +
             f64x4::splat(y2.extract(1)) * x[1] +
             f64x4::splat(y2.extract(2)) * x[2] +
             f64x4::splat(y2.extract(3)) * x[3],
             f64x4::splat(y3.extract(0)) * x[0] +
             f64x4::splat(y3.extract(1)) * x[1] +
             f64x4::splat(y3.extract(2)) * x[2] +
             f64x4::splat(y3.extract(3)) * x[3],
             ]);
        }
    })
}

#[bench]
fn inverse_naive(b: &mut B) {
    let mut x = [[0_f32; 4]; 4];
    for i in 0..4 { x[i][i] = 1.0 }

    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);

            let mut t = [[0_f32; 4]; 4];
            for i in 0..4 {
                t[0][i] = x[i][0];
                t[1][i] = x[i][1];
                t[2][i] = x[i][2];
                t[3][i] = x[i][3];
            }

            let _0 = t[2][2] * t[3][3];
            let _1 = t[2][3] * t[3][2];
            let _2 = t[2][1] * t[3][3];
            let _3 = t[2][3] * t[3][1];
            let _4 = t[2][1] * t[3][2];
            let _5 = t[2][2] * t[3][1];
            let _6 = t[2][0] * t[3][3];
            let _7 = t[2][3] * t[3][0];
            let _8 = t[2][0] * t[3][2];
            let _9 = t[2][2] * t[3][0];
            let _10 = t[2][0] * t[3][1];
            let _11 = t[2][1] * t[3][0];

            let d00 = _0 * t[1][1] + _3 * t[1][2] + _4 * t[1][3] -
                (_1 * t[1][1] + _2 * t[1][2] + _5 * t[1][3]);
            let d01 = _1 * t[1][0] + _6 * t[1][2] + _9 * t[1][3] -
                (_0 * t[1][0] + _7 * t[1][2] + _8 * t[1][3]);
            let d02 = _2 * t[1][0] + _7 * t[1][1] + _10 * t[1][3] -
                (_3 * t[1][0] + _6 * t[1][1] + _11 * t[1][3]);
            let d03 = _5 * t[1][0] + _8 * t[1][1] + _11 * t[1][2] -
                (_4 * t[1][0] + _9 * t[1][1] + _10 * t[1][2]);
            let d10 = _1 * t[0][1] + _2 * t[0][2] + _5 * t[0][3] -
                (_0 * t[0][1] + _3 * t[0][2] + _4 * t[0][3]);
            let d11 = _0 * t[0][0] + _7 * t[0][2] + _8 * t[0][3] -
                (_1 * t[0][0] + _6 * t[0][2] + _9 * t[0][3]);
            let d12 = _3 * t[0][0] + _6 * t[0][1] + _11 * t[0][3] -
                (_2 * t[0][0] + _7 * t[0][1] + _10 * t[0][3]);
            let d13 = _4 * t[0][0] + _9 * t[0][1] + _10 * t[0][2] -
                (_5 * t[0][0] + _8 * t[0][1] + _11 * t[0][2]);

            let _0 = t[0][2] * t[1][3];
            let _1 = t[0][3] * t[1][2];
            let _2 = t[0][1] * t[1][3];
            let _3 = t[0][3] * t[1][1];
            let _4 = t[0][1] * t[1][2];
            let _5 = t[0][2] * t[1][1];
            let _6 = t[0][0] * t[1][3];
            let _7 = t[0][3] * t[1][0];
            let _8 = t[0][0] * t[1][2];
            let _9 = t[0][2] * t[1][0];
            let _10 = t[0][0] * t[1][1];
            let _11 = t[0][1] * t[1][0];

            let d20  = _0*t[3][1]  + _3*t[3][2]  + _4*t[3][3]-
                (_1*t[3][1]  + _2*t[3][2]  + _5*t[3][3]);
            let d21  = _1*t[3][0]  + _6*t[3][2]  + _9*t[3][3]-
                (_0*t[3][0]  + _7*t[3][2]  + _8*t[3][3]);
            let d22 = _2*t[3][0]  + _7*t[3][1]  + _10*t[3][3]-
                (_3*t[3][0]  + _6*t[3][1]  + _11*t[3][3]);
            let d23 = _5*t[3][0]  + _8*t[3][1]  + _11*t[3][2]-
                (_4*t[3][0]  + _9*t[3][1]  + _10*t[3][2]);
            let d30 = _2*t[2][2]  + _5*t[2][3]  + _1*t[2][1]-
                (_4*t[2][3]  + _0*t[2][1]   + _3*t[2][2]);
            let d31 = _8*t[2][3]  + _0*t[2][0]   + _7*t[2][2]-
                (_6*t[2][2]  + _9*t[2][3]  + _1*t[2][0]);
            let d32 = _6*t[2][1]   + _11*t[2][3] + _3*t[2][0]-
                (_10*t[2][3] + _2*t[2][0]   + _7*t[2][1]);
            let d33 = _10*t[2][2] + _4*t[2][0]   + _9*t[2][1]-
                (_8*t[2][1]   + _11*t[2][2] + _5*t[2][0]);

            let det = t[0][0] * d00 + t[0][1] * d01 + t[0][2] * d02 + t[0][3] * d03;

            let det = 1.0 / det;
            let mut ret = [[d00, d01, d02, d03],
                           [d10, d11, d12, d13],
                           [d20, d21, d22, d23],
                           [d30, d31, d32, d33]];
            for i in 0..4 {
                for j in 0..4 {
                    ret[i][j] *= det;
                }
            }
            bb(&ret);
        }
    })
}

#[bench]
fn inverse_simd4(b: &mut B) {
    let mut x = [f32x4::splat(0_f32); 4];
    for i in 0..4 { x[i] = x[i].replace(i as u32, 1.0); }

    fn shuf0145(v: f32x4, w: f32x4) -> f32x4 {
        f32x4::new(v.extract(0), v.extract(1),
                   w.extract(4 - 4), w.extract(5 - 4))
    }
    fn shuf0246(v: f32x4, w: f32x4) -> f32x4 {
        f32x4::new(v.extract(0), v.extract(2),
                   w.extract(4 - 4), w.extract(6 - 4))
    }
    fn shuf1357(v: f32x4, w: f32x4) -> f32x4 {
        f32x4::new(v.extract(1), v.extract(3),
                   w.extract(5 - 4), w.extract(7 - 4))
    }
    fn shuf2367(v: f32x4, w: f32x4) -> f32x4 {
        f32x4::new(v.extract(2), v.extract(3),
                   w.extract(6 - 4), w.extract(7 - 4))
    }

    fn swiz1032(v: f32x4) -> f32x4 {
        f32x4::new(v.extract(1), v.extract(0),
                   v.extract(3), v.extract(2))
    }
    fn swiz2301(v: f32x4) -> f32x4 {
        f32x4::new(v.extract(2), v.extract(3),
                   v.extract(0), v.extract(1))
    }

    b.iter(|| {
        for _ in 0..100 {
            let src0;
            let src1;
            let src2;
            let src3;
            let mut tmp1;
            let row0;
            let mut row1;
            let mut row2;
            let mut row3;
            let mut minor0;
            let mut minor1;
            let mut minor2;
            let mut minor3;
            let mut det;

            let x = bb(&x);
            src0 = x[0];
            src1 = x[1];
            src2 = x[2];
            src3 = x[3];

            tmp1 = shuf0145(src0, src1);
            row1 = shuf0145(src2, src3);
            row0 = shuf0246(tmp1, row1);
            row1 = shuf1357(row1, tmp1);

            tmp1 = shuf2367(src0, src1);
            row3 = shuf2367(src2, src3);
            row2 = shuf0246(tmp1, row3);
            row3 = shuf0246(row3, tmp1);


            tmp1 = row2 * row3;
            tmp1 = swiz1032(tmp1);
            minor0 = row1 * tmp1;
            minor1 = row0 * tmp1;
            tmp1 = swiz2301(tmp1);
            minor0 = (row1 * tmp1) - minor0;
            minor1 = (row0 * tmp1) - minor1;
            minor1 = swiz2301(minor1);


            tmp1 = row1 * row2;
            tmp1 = swiz1032(tmp1);
            minor0 = (row3 * tmp1) + minor0;
            minor3 = row0 * tmp1;
            tmp1 = swiz2301(tmp1);

            minor0 = minor0 - row3 * tmp1;
            minor3 = row0 * tmp1 - minor3;
            minor3 = swiz2301(minor3);


            tmp1 = row3 * swiz2301(row1);
            tmp1 = swiz1032(tmp1);
            row2 = swiz2301(row2);
            minor0 = row2 * tmp1 + minor0;
            minor2 = row0 * tmp1;
            tmp1 = swiz2301(tmp1);
            minor0 = minor0 - row2 * tmp1;
            minor2 = row0 * tmp1 - minor2;
            minor2 = swiz2301(minor2);


            tmp1 = row0 * row1;
            tmp1 = swiz1032(tmp1);
            minor2 = minor2 + row3 * tmp1;
            minor3 = row2 * tmp1 - minor3;
            tmp1 = swiz2301(tmp1);
            minor2 = row3 * tmp1 - minor2;
            minor3 = minor3 - row2 * tmp1;



            tmp1 = row0 * row3;
            tmp1 = swiz1032(tmp1);
            minor1 = minor1 - row2 * tmp1;
            minor2 = row1 * tmp1 + minor2;
            tmp1 = swiz2301(tmp1);
            minor1 = row2 * tmp1 + minor1;
            minor2 = minor2 - row1 * tmp1;

            tmp1 = row0 * row2;
            tmp1 = swiz1032(tmp1);
            minor1 = row3 * tmp1 + minor1;
            minor3 = minor3 - row1 * tmp1;
            tmp1 = swiz2301(tmp1);
            minor1 = minor1 - row3 * tmp1;
            minor3 = row1 * tmp1 + minor3;

            det = row0 * minor0;
            det = swiz2301(det) + det;
            det = swiz1032(det) + det;
            //tmp1 = det.approx_reciprocal(); det = tmp1 * (f32x4::splat(2.0) - det * tmp1);
            det = f32x4::splat(1.0) / det;

            bb(&[minor0 * det, minor1 * det, minor2 * det, minor3 * det]);
        }
     })

}

#[cfg(target_feature = "avx")]
#[bench]
fn inverse_simd8(b: &mut B) {
    use simd::x86::sse3::Sse3F32x4;
    use simd::x86::avx::{f32x8, LowHigh128, AvxF32x8};

    let mut x = [f32x4::splat(0_f32); 4];
    for i in 0..4 { x[i] = x[i].replace(i as u32, 1.0); }

    let x = [from_halves(x[0], x[1]), from_halves(x[2], x[3])];

    // The following functions simplify to single AVX/AVX2 instructions
    fn from_halves(low: f32x4, high: f32x4) -> f32x8 {
        f32x8::new(low.extract(0), low.extract(1), low.extract(2), low.extract(3),
                   high.extract(0), high.extract(1), high.extract(2), high.extract(3))
     }

    fn rot1(x: f32x8) -> f32x8 {
        f32x8::new(x.extract(1), x.extract(2), x.extract(3), x.extract(0),
                   x.extract(5), x.extract(6), x.extract(7), x.extract(4))
    }

    fn rot2(x: f32x8) -> f32x8 {
        rot1(rot1(x))
    }

    fn swap_lanes(x: f32x8) -> f32x8 {
        f32x8::new(x.extract(4), x.extract(5), x.extract(6), x.extract(7),
            x.extract(0), x.extract(1), x.extract(2), x.extract(3))
    }

    fn unpack_lo_pd(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(0), v.extract(1), w.extract(0), w.extract(1),
                   v.extract(4), v.extract(5), w.extract(4), w.extract(5),)
    }
    fn unpack_hi_pd(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(2), v.extract(3), w.extract(2), w.extract(3),
                   v.extract(6), v.extract(7), w.extract(6), w.extract(7),)
    }

    fn rot_shuf(x: f32x8) -> f32x8 {
        f32x8::new(x.extract(5), x.extract(1), x.extract(7), x.extract(3),
                   x.extract(6), x.extract(2), x.extract(4), x.extract(0))
    }

    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);
            let x01 = x[0];
            let x23 = x[1];
            let x3 = x23.high();

            let x10 = swap_lanes(x01);
            let x10r1 = rot1(x10);
            let x10r2 = rot2(x10);

            let x32 = swap_lanes(x23);
            let x32r1 = rot1(x32);
            let x32r2 = rot2(x32);

            // multiply rotated rows to create adjugate of A
            // adj(A) ~ reverse_rows([rot1(y32), rot1(y10)]) (up to signs)
            let p01_23r2 = x01 * rot2(x23);
            let y10 = p01_23r2 * x32r1;
            let y32 = rot2(p01_23r2) * x10r1;

            let y32 = y32 - p01_23r2 * x10r1;
            let y10 = y10 - rot2(p01_23r2) * x32r1;

            let p01_23r1 = x01 * rot1(x23);
            let y32 = y32 + p01_23r1 * x10r2;
            let y10 = y10 - p01_23r1 * x32r2;

            let y32 = y32 + rot1(p01_23r1) * x10;
            let y10 = y10 - rot1(p01_23r1) * x32;

            let p01r1_23 = rot1(x01) * x23;
            let y32 = y32 - p01r1_23 * x10r2;
            let y10 = y10 + p01r1_23 * x32r2;

            let y32 = y32 - rot1(p01r1_23) * x10;
            let y10 = y10 + rot1(p01r1_23) * x32;

            let det = rot1(y32).low() * x3;
            let det = det.hsub(det);
            let det = det.hadd(det);
            // need to fix signs
            let det_rcp = f32x4::new(-1.0, 1.0, -1.0, 1.0) / det;
            let det_rcp = from_halves(det_rcp, det_rcp);

            // multiplying here fixes the signs
            let y32 = y32 * det_rcp;
            let y10 = y10 * det_rcp;

            // rotate and transpose the adjugate
            let z32 = rot_shuf(y32);
            let z10 = rot_shuf(y10);
            let r01 = unpack_lo_pd(z10, z32);
            let r23 = unpack_hi_pd(z10, z32);

            bb(&[r01, r23]);
        }
     })

}

#[bench]
fn transpose_naive(b: &mut B) {
    let x = [[0_f32; 4]; 4];
    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);
            bb(&[[x[0][0], x[1][0], x[2][0], x[3][0]],
                 [x[0][1], x[1][1], x[2][1], x[3][1]],
                 [x[0][2], x[1][2], x[2][2], x[3][2]],
                 [x[0][3], x[1][3], x[2][3], x[3][3]]]);
        }
    })
}

#[bench]
fn transpose_simd4(b: &mut B) {
    let x = [f32x4::splat(0_f32); 4];

    fn shuf0246(v: f32x4, w: f32x4) -> f32x4 {
        f32x4::new(v.extract(0), v.extract(2),
                   w.extract(4 - 4), w.extract(6 - 4))
    }
    fn shuf1357(v: f32x4, w: f32x4) -> f32x4 {
        f32x4::new(v.extract(1), v.extract(3),
                   w.extract(5 - 4), w.extract(7 - 4))
    }
    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);
            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];
            let x3 = x[3];

            let a0 = shuf0246(x0, x1);
            let a1 = shuf0246(x2, x3);
            let a2 = shuf1357(x0, x1);
            let a3 = shuf1357(x2, x3);

            let b0 = shuf0246(a0, a1);
            let b1 = shuf0246(a2, a3);
            let b2 = shuf1357(a0, a1);
            let b3 = shuf1357(a2, a3);
            bb(&[b0, b1, b2, b3]);
        }
    })
}

#[cfg(target_feature = "avx")]
#[bench]
fn transpose_simd8_naive(b: &mut B) {
    let x = [f32x8::splat(0_f32); 2];

    fn shuf0246(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(0), v.extract(2), v.extract(4), v.extract(6),
                   w.extract(0), w.extract(2), w.extract(4), w.extract(6))
    }
    fn shuf1357(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(1), v.extract(3), v.extract(5), v.extract(7),
                   w.extract(1), w.extract(3), w.extract(5), w.extract(7),)
    }
    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);
            let x01 = x[0];
            let x23 = x[1];

            let a01 = shuf0246(x01, x23);
            let a23 = shuf1357(x01, x23);

            let b01 = shuf0246(a01, a23);
            let b23 = shuf1357(a01, a23);
            bb(&[b01, b23]);
        }
    })
}

#[cfg(target_feature = "avx")]
#[bench]
fn transpose_simd8_avx2_vpermps(b: &mut B) {
    let x = [f32x8::splat(0_f32); 2];

    // efficient on AVX2 using vpermps
    fn perm04152637(v: f32x8) -> f32x8 {
        // broken on rustc 1.7.0-nightly (1ddaf8bdf 2015-12-12)
        // v.permutevar(i32x8::new(0, 4, 1, 5, 2, 6, 3, 7))
        f32x8::new(v.extract(0), v.extract(4), v.extract(1), v.extract(5),
                    v.extract(2), v.extract(6), v.extract(3), v.extract(7))
    }
    fn shuf_lo(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(0), v.extract(1), w.extract(0), w.extract(1),
                   v.extract(4), v.extract(5), w.extract(4), w.extract(5),)
    }
    fn shuf_hi(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(2), v.extract(3), w.extract(2), w.extract(3),
                   v.extract(6), v.extract(7), w.extract(6), w.extract(7),)
    }
    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);
            let x01 = x[0];
            let x23 = x[1];

            let a01 = perm04152637(x01);
            let a23 = perm04152637(x23);

            let b01 = shuf_lo(a01, a23);
            let b23 = shuf_hi(a01, a23);
            bb(&[b01, b23]);
        }
    })
}

#[cfg(target_feature = "avx")]
#[bench]
fn transpose_simd8_avx2_vpermpd(b: &mut B) {
    let x = [f32x8::splat(0_f32); 2];

    // efficient on AVX2 using vpermpd
    fn perm01452367(v: f32x8) -> f32x8 {
        f32x8::new(v.extract(0), v.extract(1), v.extract(4), v.extract(5),
                    v.extract(2), v.extract(3), v.extract(6), v.extract(7))
    }
    fn shuf_lo_ps(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(0), w.extract(0), v.extract(1), w.extract(1),
                   v.extract(4), w.extract(4), v.extract(5), w.extract(5),)
    }
    fn shuf_hi_ps(v: f32x8, w: f32x8) -> f32x8 {
        f32x8::new(v.extract(2), w.extract(2), v.extract(3), w.extract(3),
                   v.extract(6), w.extract(6), v.extract(7), w.extract(7),)
    }
    b.iter(|| {
        for _ in 0..100 {
            let x = bb(&x);
            let x01 = x[0];
            let x23 = x[1];

            let a01 = perm01452367(x01);
            let a23 = perm01452367(x23);

            let b01 = shuf_lo_ps(a01, a23);
            let b23 = shuf_hi_ps(a01, a23);
            bb(&[b01, b23]);
        }
    })
}
