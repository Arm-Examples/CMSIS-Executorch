#ifndef IMU_H_
#define IMU_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Starts the BMI323 accelerometer + gyroscope. Returns 0 on success. */
int imu_init(void);

/* Reads the latest sample. *motion = gyro deg/s + 100 x change in g since the
 * last call (about 0-5 when the board sits still). ax/ay/az (in g) may be NULL.
 * Returns 0 on success. */
int imu_poll(float *motion, float *ax, float *ay, float *az);

/* Gyro (deg/s, x y z) from the latest imu_poll() */
void imu_last_gyro(float *g);

#ifdef __cplusplus
}
#endif

#endif /* IMU_H_ */
