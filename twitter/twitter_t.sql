/*
 Navicat Premium Data Transfer

 Source Server         : localhost80
 Source Server Type    : MySQL
 Source Server Version : 80033
 Source Host           : 127.0.0.1:3307
 Source Schema         : test

 Target Server Type    : MySQL
 Target Server Version : 80033
 File Encoding         : 65001

 Date: 07/07/2023 20:00:12
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for twitter_t
-- ----------------------------
DROP TABLE IF EXISTS `twitter_t`;
CREATE TABLE `twitter_t`  (
  `user_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `content` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `comment_num` int(0) NULL DEFAULT NULL,
  `transmitting_num` int(0) NULL DEFAULT NULL,
  `approval_num` int(0) NULL DEFAULT NULL,
  `view_num` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
