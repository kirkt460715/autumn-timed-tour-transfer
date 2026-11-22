# 资料约定

CSV 采用 UTF-8 编码。日期使用年、月、日，数值以十进制表示。字段为空表示来源未提供该项或该项不适用，含义以表约定为准。

时刻均为当地时间，时分字段采用 HH:MM。相邻段具有相同 `day` 和 `corridor` 时共享观测日背景，旅程段通过相同 trip 组合。结束签到上限含该分钟，家庭一票同行。只有运营书面登记为 `exclusive` 的触发才能共享应急席，它们以共同的 group 表示互斥，并以 deadline 作为去留截止。capacity 是可容人数。accessible=no 的班次不能交给需要无障碍换乘的家庭。reserve 中 locked 是已对客承诺。
线路门槛针对全部场段均可完成的家庭占比，按 `family_route` 中的最小可接受概率检查。运营登记的无车步行时间为分钟，车开停复核按通知的生效时刻而非收件顺序。

## 文件来源

- `slots.csv`。场馆时段与可换空位。 记录键为 `slot`。
- `rides.csv`。共享班次与整组可用席数。 记录键为 `ride`。
- `walks.csv`。不占车位的公开线路。 记录键为 `corridor`、`allowed_group`。
- `family_route.csv`。已售线路的整组人数、准点约定和各馆可换时段。floor 为零到一。 记录键为 `family`。
- `reserves.csv`。已收家庭通知的席位承诺。 记录键为 `reservation`。
- `activation.csv`。备用席的业务触发条件。 记录键为 `trigger`。
- `notices.csv`。运营通知。 记录键为 `notice`。
- `observations.csv`。同一历史观察日中的结束、搭车和停开记录。 记录键为 `record`。
