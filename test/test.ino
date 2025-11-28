const int servoPin = 9;  // 舵机信号引脚

void setup() {
  Serial.begin(9600);
  pinMode(servoPin, OUTPUT);
  
  // 初始化到0度位置
  moveToAngle(0);
  Serial.println("Ready: Send '0' for 0°, '1' for 45°");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '1') {
      moveToAngle(45);  // 转到45度
    }
    else if (c == '0') {
      moveToAngle(0);   // 回到0度
    }
  }
}

// 脉冲函数
void servoPulse(int angle) {
  // 将角度转换为脉宽 (0°=500us, 180°=2500us)
  int pulseWidth = map(angle, 0, 180, 500, 2500);
  pulseWidth = constrain(pulseWidth, 500, 2500);
  
  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulseWidth);
  digitalWrite(servoPin, LOW);
  delayMicroseconds(20000 - pulseWidth); // 补足20ms周期
}

// 移动到指定角度
void moveToAngle(int angle) {
  // 发送50个脉冲确保舵机稳定到达位置
  for(int i = 0; i < 50; i++) {
    servoPulse(angle);
  }
}