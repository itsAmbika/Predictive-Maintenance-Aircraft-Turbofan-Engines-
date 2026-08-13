import { useEffect, useRef } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";

export function AnimatedNumber({
  value,
  decimals = 0,
  suffix = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
}) {
  const motionValue = useMotionValue(0);
  const spring = useSpring(motionValue, { stiffness: 90, damping: 20, mass: 0.6 });
  const rounded = useTransform(spring, (v) => `${v.toFixed(decimals)}${suffix}`);
  const first = useRef(true);

  useEffect(() => {
    motionValue.set(first.current ? 0 : motionValue.get());
    first.current = false;
    motionValue.set(value);
  }, [value, motionValue]);

  return <motion.span>{rounded}</motion.span>;
}
